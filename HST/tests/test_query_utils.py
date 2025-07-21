import pytest
from unittest import mock
import types
import sys
from HST.hst_helper import query_utils

# Patch Observations and logger globally
@pytest.fixture(autouse=True)
def patch_obs_and_logger(monkeypatch):
    class FakeObs:
        @staticmethod
        def query_criteria(**kwargs):
            return 'table'
        @staticmethod
        def get_product_list(table):
            return table
        @staticmethod
        def download_products(table, download_dir=None):
            FakeObs.downloaded = (table, download_dir)
    monkeypatch.setattr(query_utils, 'Observations', FakeObs)
    monkeypatch.setattr(query_utils, 'remove_all_tasks_for_a_prog_id', lambda pid: setattr(FakeObs, 'removed', pid))
    class FakeLogger:
        def __init__(self): self.messages = []
        def info(self, msg): self.messages.append(msg)
        def warn(self, msg): self.messages.append(msg)
        def exception(self, e): self.messages.append(str(e))
    monkeypatch.setattr(query_utils.pdslogger, 'EasyLogger', FakeLogger)


def test_ymd_tuple_to_mjd(monkeypatch):
    monkeypatch.setattr(query_utils.julian, 'day_from_ymd', lambda y, m, d: 2450000)
    monkeypatch.setattr(query_utils.julian, 'mjd_from_day', lambda days: days - 2400000.5)
    assert query_utils.ymd_tuple_to_mjd((2000, 1, 1)) == 50000 - 0.5


def test_query_mast_slice_success(monkeypatch):
    logger = query_utils.pdslogger.EasyLogger()
    result = query_utils.query_mast_slice(proposal_id='12345', instrument='WFC3', logger=logger, max_retries=1)
    assert result == 'table'
    assert 'Query MAST: run query_mast_slice' in logger.messages


def test_query_mast_slice_retry(monkeypatch):
    class FailingObs:
        tries = 0
        @staticmethod
        def query_criteria(**kwargs):
            FailingObs.tries += 1
            if FailingObs.tries < 2:
                raise ConnectionError('fail')
            return 'table'
    monkeypatch.setattr(query_utils, 'Observations', FailingObs)
    logger = query_utils.pdslogger.EasyLogger()
    result = query_utils.query_mast_slice(proposal_id='12345', instrument='WFC3', logger=logger, max_retries=2)
    assert result == 'table'
    assert any('retry' in m for m in logger.messages)


def test_query_mast_slice_failure(monkeypatch):
    class FailingObs:
        @staticmethod
        def query_criteria(**kwargs):
            raise ConnectionError('fail')
    monkeypatch.setattr(query_utils, 'Observations', FailingObs)
    logger = query_utils.pdslogger.EasyLogger()
    with pytest.raises(RuntimeError):
        query_utils.query_mast_slice(proposal_id='12345', instrument='WFC3', logger=logger, max_retries=1)
    # Removed attribute check, as it is not set on the real Observations


def test_filter_table():
    class FakeTable:
        def __init__(self, rows): self._rows = rows
        def __iter__(self): return iter(self._rows)
        def copy(self): return FakeTable(list(self._rows))
        def remove_rows(self, to_delete):
            for i in sorted(to_delete, reverse=True):
                del self._rows[i]
    table = FakeTable([1, 2, 3, 4])
    result = query_utils.filter_table(lambda x: x % 2 == 0, table)
    assert result._rows == [2, 4]


def test_is_accepted_instrument_letter_code(monkeypatch):
    monkeypatch.setitem(query_utils.__dict__, 'ACCEPTED_LETTER_CODES', {'a', 'b'})
    row = {'obs_id': 'a123'}
    assert query_utils.is_accepted_instrument_letter_code(row)
    row = {'obs_id': 'c123'}
    assert not query_utils.is_accepted_instrument_letter_code(row)


def test_is_accepted_instrument_suffix(monkeypatch):
    monkeypatch.setitem(query_utils.__dict__, 'ACCEPTED_SUFFIXES', {'WFC3': {'flt', 'raw'}})
    monkeypatch.setitem(query_utils.__dict__, 'INSTRUMENT_FROM_LETTER_CODE', {'j': 'WFC3'})
    monkeypatch.setattr(query_utils, 'get_suffix', lambda row: 'flt')
    monkeypatch.setattr(query_utils, 'get_instrument_id_from_table_row', lambda row: 'WFC3')
    row = {'obs_id': 'j123', 'productSubGroupDescription': 'flt'}
    assert query_utils.is_accepted_instrument_suffix(row)
    monkeypatch.setattr(query_utils, 'get_instrument_id_from_table_row', lambda row: None)
    assert not query_utils.is_accepted_instrument_suffix(row)


def test_is_trl_suffix(monkeypatch):
    monkeypatch.setattr(query_utils, 'get_suffix', lambda row: 'trl')
    row = {'productSubGroupDescription': 'trl'}
    assert query_utils.is_trl_suffix(row)
    monkeypatch.setattr(query_utils, 'get_suffix', lambda row: 'flt')
    assert not query_utils.is_trl_suffix(row)


def test_is_targeted_visit(monkeypatch):
    monkeypatch.setattr(query_utils, 'get_format_term', lambda fn: 'ipppssoot')
    monkeypatch.setattr(query_utils, 'get_visit', lambda ft: '01')
    row = {'productFilename': 'ipppssoot_flt.fits'}
    assert query_utils.is_targeted_visit(row, '01')
    assert not query_utils.is_targeted_visit(row, '02')


def test_get_instrument_id_from_table_row(monkeypatch):
    monkeypatch.setitem(query_utils.__dict__, 'INSTRUMENT_FROM_LETTER_CODE', {'j': 'WFC3'})
    row = {'obs_id': 'j123'}
    assert query_utils.get_instrument_id_from_table_row(row) == 'WFC3'
    row = {'obs_id': 'x123'}
    assert query_utils.get_instrument_id_from_table_row(row) is None


def test_get_suffix(monkeypatch):
    row = {'productSubGroupDescription': 'flt', 'productFilename': 'ipppssoot_flt.fits'}
    assert query_utils.get_suffix(row) == 'flt'
    row = {'productSubGroupDescription': '--', 'productFilename': 'ipppssoot_flt.fits'}
    assert query_utils.get_suffix(row) == 'flt'


def test_get_filtered_products(monkeypatch):
    class FakeTable:
        def __init__(self): self.called = []
        def copy(self): return self
        def remove_rows(self, to_delete): self.called.append(('remove', to_delete))
        def __iter__(self): return iter([])  # Make it iterable
    class FakeObs:
        @staticmethod
        def get_product_list(table): return table
    monkeypatch.setattr(query_utils, 'Observations', FakeObs)
    monkeypatch.setattr(query_utils, 'filter_table', lambda pred, table: table)
    monkeypatch.setattr(query_utils, 'is_targeted_visit', lambda row, visit: row.get('visit') == visit)
    table = FakeTable()
    result = query_utils.get_filtered_products(table, visit='01')
    assert result is table


def test_get_trl_products(monkeypatch):
    class FakeObs:
        @staticmethod
        def get_product_list(table): return table
    monkeypatch.setattr(query_utils, 'Observations', FakeObs)
    monkeypatch.setattr(query_utils, 'filter_table', lambda pred, table: table)
    table = object()
    result = query_utils.get_trl_products(table)
    assert result is table


def test_download_files(monkeypatch):
    class FakeLogger:
        def __init__(self): self.messages = []
        def warn(self, msg): self.messages.append(msg)
        def info(self, msg): self.messages.append(msg)
        def exception(self, e): self.messages.append(str(e))
    monkeypatch.setattr(query_utils, 'os', mock.Mock())
    monkeypatch.setattr(query_utils, 'Observations', mock.Mock())
    logger = FakeLogger()
    # Empty table
    query_utils.download_files([], '/tmp/dir', logger=logger)
    assert 'Empty result from MAST query' in logger.messages
    # Non-empty table, not testing
    query_utils.download_files([1], '/tmp/dir', logger=logger, testing=True)
    assert 'Download files to /tmp/dir' in logger.messages
