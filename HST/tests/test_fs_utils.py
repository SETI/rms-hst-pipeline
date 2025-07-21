import os
import shutil
import tempfile
import builtins
import pytest
from unittest import mock
from HST.hst_helper import fs_utils

# Mock HST_DIR for all tests
@pytest.fixture(autouse=True)
def patch_hst_dir(monkeypatch):
    monkeypatch.setitem(fs_utils.__dict__, 'HST_DIR', {'staging': '/tmp/staging', 'pipeline': '/tmp/pipeline', 'bundles': '/tmp/bundles'})


def test_create_program_dir_creates_directory(monkeypatch):
    called = {}
    def fake_makedirs(path, exist_ok):
        called['path'] = path
        called['exist_ok'] = exist_ok
    monkeypatch.setattr(os, 'makedirs', fake_makedirs)
    monkeypatch.setattr(fs_utils, 'get_program_dir_path', lambda *a, **k: '/tmp/pipeline/hst_12345')
    result = fs_utils.create_program_dir('12345')
    assert result == '/tmp/pipeline/hst_12345'
    assert called['path'] == '/tmp/pipeline/hst_12345'
    assert called['exist_ok'] is True


def test_get_program_dir_path_formats(monkeypatch):
    monkeypatch.setitem(fs_utils.__dict__, 'HST_DIR', {'pipeline': '/tmp/pipeline', 'bundles': '/tmp/bundles', 'staging': '/tmp/staging'})
    monkeypatch.setattr(fs_utils, 'get_formatted_proposal_id', lambda pid: str(pid).zfill(5))
    # No visit
    assert fs_utils.get_program_dir_path('123', None, 'pipeline') == '/tmp/pipeline/hst_00123'
    # With visit
    assert fs_utils.get_program_dir_path('123', '01', 'pipeline') == '/tmp/pipeline/hst_00123/visit_01'
    # Testing flag
    assert fs_utils.get_program_dir_path('123', None, 'pipeline', testing=True) == '/tmp/pipeline/hst_00123_testing'


def test_get_deliverable_path(monkeypatch):
    monkeypatch.setattr(fs_utils, 'get_formatted_proposal_id', lambda pid: str(pid).zfill(5))
    monkeypatch.setattr(fs_utils, 'get_program_dir_path', lambda pid, v, r, testing=False: f'/tmp/bundles/hst_{str(pid).zfill(5)}')
    result = fs_utils.get_deliverable_path('12345')
    assert result == '/tmp/bundles/hst_12345/hst_12345-deliverable'


def test_get_format_term():
    assert fs_utils.get_format_term('ipppssoot_suffix.fits') == 'ipppssoot'
    assert fs_utils.get_format_term('abc_def_ghi.txt') == 'abc'


def test_get_instrument_id_from_fname(monkeypatch):
    monkeypatch.setitem(fs_utils.__dict__, 'INSTRUMENT_FROM_LETTER_CODE', {'j': 'WFC3', 'h': 'ACS'})
    assert fs_utils.get_instrument_id_from_fname('j12345abc.fits') == 'WFC3'
    assert fs_utils.get_instrument_id_from_fname('h54321xyz.fits') == 'ACS'
    assert fs_utils.get_instrument_id_from_fname('x00000.fits') is None


def test_get_file_suffix():
    assert fs_utils.get_file_suffix('ipppssoot_suffix.fits') == 'suffix'
    assert fs_utils.get_file_suffix('abc_def_ghi.txt') == 'def_ghi'  # The function returns everything after the first underscore


def test_get_visit():
    assert fs_utils.get_visit('ipppssoot') == 'ss'  # The function returns index 4:6
    assert fs_utils.get_visit('abcdefgh') == 'ef'


def test_file_md5(tmp_path):
    file = tmp_path / 'file.txt'
    file.write_bytes(b'hello world')
    result = fs_utils.file_md5(str(file))
    import hashlib
    expected = hashlib.md5(b'hello world').hexdigest()
    assert result == expected


def test_get_formatted_proposal_id():
    assert fs_utils.get_formatted_proposal_id(123) == '00123'
    assert fs_utils.get_formatted_proposal_id('42') == '00042'
    assert fs_utils.get_formatted_proposal_id(12345) == '12345'


def test_backup_file(monkeypatch, tmp_path):
    called = {}
    def fake_makedirs(path, exist_ok):
        called['makedirs'] = path
    def fake_move(src, dst):
        called['move'] = (src, dst)
    monkeypatch.setattr(os, 'makedirs', fake_makedirs)
    monkeypatch.setattr(shutil, 'move', fake_move)
    monkeypatch.setattr(fs_utils, 'get_program_dir_path', lambda pid, visit: str(tmp_path))
    now = '2022-01-01T00-00-00'
    monkeypatch.setattr(fs_utils.datetime, 'datetime', mock.Mock(now=lambda: mock.Mock(strftime=lambda fmt: now)))
    monkeypatch.setattr(fs_utils, 'get_formatted_proposal_id', lambda pid: str(pid).zfill(5))
    test_file = tmp_path / 'testfile.txt'
    test_file.write_text('data')
    fs_utils.backup_file('12345', '01', str(test_file))
    assert 'makedirs' in called
    assert 'move' in called
    assert called['makedirs'].endswith('/backups')
    assert called['move'][0] == str(test_file)
    assert called['move'][1].endswith('.txt')


def test_create_col_dir_in_bundle(monkeypatch):
    called = {}
    def fake_makedirs(path, exist_ok):
        called['path'] = path
    monkeypatch.setattr(os, 'makedirs', fake_makedirs)
    monkeypatch.setattr(fs_utils, 'get_deliverable_path', lambda pid, testing=False: '/tmp/bundles/hst_12345/hst_12345-deliverable')
    result = fs_utils.create_col_dir_in_bundle('12345', 'data_collection')
    assert result[0] == '/tmp/bundles/hst_12345/hst_12345-deliverable'
    assert result[1] == '/tmp/bundles/hst_12345/hst_12345-deliverable/data_collection'
    assert called['path'].endswith('data_collection')
