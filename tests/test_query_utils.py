import pytest
from unittest import mock
import types
import sys
from astropy.table import Table
from hst_pipeline.hst_helper import query_utils


def make_obs_table():
    """An observation table mixing moving and fixed targets of several instruments."""
    return Table({'obs_id': ['w1', 'w2', 'f1', 'h1', 's1'],
                  'instrument_name': ['WFC3/UVIS', 'WFC3/IR', 'FGS', 'HSP/UNK/VIS',
                                      'STIS/CCD'],
                  'mtFlag': [True, False, False, False, False]})

# Patch Observations and logger globally
@pytest.fixture(autouse=True)
def patch_obs_and_logger(monkeypatch):
    class FakeObs:
        @staticmethod
        def query_criteria(**kwargs):
            FakeObs.kwargs = kwargs
            return make_obs_table()
        @staticmethod
        def get_product_list(table):
            return table
        @staticmethod
        def download_products(table, download_dir=None):
            FakeObs.downloaded = (table, download_dir)
    monkeypatch.setattr(query_utils, 'Observations', FakeObs)
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
    assert list(result['obs_id']) == ['w1', 'f1', 'h1']
    assert 'Query MAST: run query_mast_slice' in logger.messages


def test_release_range_comes_from_end_date():
    # tests/conftest.py limits the tests to data released before 2025.
    from hst_pipeline.hst_helper import END_DATE, START_DATE
    assert END_DATE == (2025, 1, 1)
    query_utils.query_mast_slice(proposal_id='17739', max_retries=1)
    start, end = query_utils.Observations.kwargs['t_obs_release']
    assert start == query_utils.ymd_tuple_to_mjd(START_DATE)
    assert end == query_utils.ymd_tuple_to_mjd((2025, 1, 1))


def test_query_mast_slice_proposal_keeps_fgs_and_hsp():
    # For a proposal, MAST is queried without mtFlag; moving targets plus every FGS
    # and HSP observation are kept.
    result = query_utils.query_mast_slice(proposal_id='8105', max_retries=1)
    assert 'mtFlag' not in query_utils.Observations.kwargs
    assert query_utils.Observations.kwargs['proposal_id'] == '08105'
    assert list(result['obs_id']) == ['w1', 'f1', 'h1']


def test_query_mast_slice_without_proposal_requires_moving_target():
    # A broad query still asks MAST for moving targets only, and is not filtered here.
    result = query_utils.query_mast_slice(instrument='FGS', max_retries=1)
    assert query_utils.Observations.kwargs['mtFlag'] is True
    assert len(result) == 5


def test_is_retrievable_row():
    rows = {row['obs_id']: row for row in make_obs_table()}
    assert query_utils.is_retrievable_row(rows['w1'])
    assert not query_utils.is_retrievable_row(rows['w2'])
    assert query_utils.is_retrievable_row(rows['f1'])
    assert query_utils.is_retrievable_row(rows['h1'])
    assert not query_utils.is_retrievable_row(rows['s1'])
    masked = Table({'instrument_name': ['FOS/BL'], 'mtFlag': [True]}, masked=True)
    masked['mtFlag'].mask = [True]
    assert not query_utils.is_retrievable_row(masked[0])


def test_query_mast_slice_retry(monkeypatch):
    class FailingObs:
        tries = 0
        @staticmethod
        def query_criteria(**kwargs):
            FailingObs.tries += 1
            if FailingObs.tries < 2:
                raise ConnectionError('fail')
            return make_obs_table()
    monkeypatch.setattr(query_utils, 'Observations', FailingObs)
    logger = query_utils.pdslogger.EasyLogger()
    result = query_utils.query_mast_slice(proposal_id='12345', instrument='WFC3', logger=logger, max_retries=2)
    assert len(result) == 3
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
    row = {'obs_id': 'j123', 'productSubGroupDescription': 'flt',
           'productFilename': 'j123_flt.fits'}
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


def test_tmp_download_filename():
    row = {'productFilename': 'j8hp01a1q_trl.fits'}
    assert query_utils.tmp_download_filename(row) == 'tmp_j8hp01a1q_trl.fits'


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
    assert 'Downloading files to /tmp/dir' in logger.messages


def test_download_files_overwrites_existing_with_filename_fn(tmp_path, monkeypatch):
    calls = []

    def fake_download_file(uri, *, local_path=None, cache=True, cloud_only=False):
        calls.append((uri, local_path, cache))
        with open(local_path, 'w') as f:
            f.write('new')
        return 'COMPLETE', None, None

    class FakeObs:
        download_file = staticmethod(fake_download_file)

    monkeypatch.setattr(query_utils, 'Observations', FakeObs)

    row = {
        'obs_collection': 'HST',
        'obs_id': 'obs1',
        'productFilename': 'file.trl',
        'dataURI': 'mast:HST/product/file.trl',
    }
    product_dir = tmp_path / 'mastDownload' / 'HST' / 'obs1'
    product_dir.mkdir(parents=True)
    existing = product_dir / 'tmp_file.trl'
    existing.write_text('old')

    query_utils.download_files(
        [row], str(tmp_path), filename_fn=lambda r: f'tmp_{r["productFilename"]}'
    )

    assert existing.read_text() == 'new'
    assert calls == [('mast:HST/product/file.trl', str(existing), False)]


def test_rename_tmp_prefixed_downloads_overwrites_existing(tmp_path):
    base = tmp_path / 'mastDownload' / 'HST' / 'obs1'
    base.mkdir(parents=True)
    (base / 'file.trl').write_text('old')
    (base / 'tmp_file.trl').write_text('new')
    query_utils.rename_tmp_prefixed_downloads(str(tmp_path))
    assert not (base / 'tmp_file.trl').exists()
    assert (base / 'file.trl').read_text() == 'new'


def test_rename_tmp_prefixed_downloads(tmp_path):
    base = tmp_path / 'mastDownload' / 'HST' / 'obs1'
    base.mkdir(parents=True)
    src = base / 'tmp_file.trl'
    src.write_text('data')
    query_utils.rename_tmp_prefixed_downloads(str(tmp_path))
    assert not src.exists()
    assert (base / 'file.trl').read_text() == 'data'


def test_rename_tmp_prefixed_downloads_skips_non_prefixed(tmp_path):
    base = tmp_path / 'mastDownload' / 'HST' / 'obs1'
    base.mkdir(parents=True)
    f = base / 'file.trl'
    f.write_text('data')
    query_utils.rename_tmp_prefixed_downloads(str(tmp_path))
    assert f.read_text() == 'data'
