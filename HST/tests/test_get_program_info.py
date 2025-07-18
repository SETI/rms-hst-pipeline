##########################################################################################
# tests/test_get_program_info.py
#
# Tests related to get_prgroam_info task
##########################################################################################

import pytest
import shutil
import tempfile
import os
from unittest import mock
import types
import urllib.error
from socket import timeout

from get_program_info import (
    get_program_info,
    is_proposal_file_retrieved,
    is_proposal_file_different,
    create_program_info_file,
    download_proposal_files
)

class TestGetProgramInfo:
    def setup_method(self):
        self.download_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.download_dir)

    @pytest.mark.parametrize(
        'p_id,expected',
        [
            ('7885', {"07885.pro", "07885.prop"}),
            ('9748', {"09748.apt", "09748.pro"}),
        ],
    )
    def test_get_program_info(self, p_id, expected, monkeypatch):
        monkeypatch.setattr('get_program_info.download_proposal_files', lambda pid, ddir, logger: expected)
        res = get_program_info(p_id, self.download_dir)
        assert expected == res

    def test_is_proposal_file_retrieved_success(self, monkeypatch, tmp_path):
        class FakeResp:
            def read(self):
                return b'newdata'
        monkeypatch.setattr('urllib.request.urlopen', lambda url, timeout: FakeResp())
        monkeypatch.setattr('get_program_info.is_proposal_file_different', lambda new, fp: True)
        monkeypatch.setattr('os.path.exists', lambda fp: False)
        m = mock.mock_open()
        monkeypatch.setattr('builtins.open', m)
        result = is_proposal_file_retrieved(12345, 'url', str(tmp_path / 'file.pro'), logger=None)
        assert result is True

    def test_is_proposal_file_retrieved_urlerror(self, monkeypatch):
        monkeypatch.setattr('urllib.request.urlopen', mock.Mock(side_effect=mock.Mock(side_effect=urllib.error.URLError(timeout('timeout')))))
        result = is_proposal_file_retrieved(12345, 'url', 'file.pro', logger=None)
        assert result is False

    def test_is_proposal_file_retrieved_urlerror_other(self, monkeypatch):
        class FakeReason(Exception):
            pass
        monkeypatch.setattr('urllib.request.urlopen', mock.Mock(side_effect=urllib.error.URLError(FakeReason())))
        result = is_proposal_file_retrieved(12345, 'url', 'file.pro', logger=None)
        assert result is False

    def test_is_proposal_file_retrieved_no_diff_creates_program_info(self, monkeypatch, tmp_path):
        class FakeResp:
            def read(self):
                return b'newdata'
        monkeypatch.setattr('urllib.request.urlopen', lambda url, timeout: FakeResp())
        monkeypatch.setattr('get_program_info.is_proposal_file_different', lambda new, fp: False)
        monkeypatch.setattr('os.path.exists', lambda fp: False)
        called = {}
        monkeypatch.setattr('get_program_info.create_program_info_file', lambda fp: called.setdefault('called', True))
        result = is_proposal_file_retrieved(12345, 'url', str(tmp_path / 'file.pro'), logger=None)
        assert result is False
        assert called['called']

    def test_is_proposal_file_different_true_false(self, tmp_path):
        file_path = tmp_path / 'file.pro'
        file_path.write_bytes(b'old')
        assert is_proposal_file_different(b'new', str(file_path)) is True
        assert is_proposal_file_different(b'old', str(file_path)) is False
        assert is_proposal_file_different(b'anything', str(tmp_path / 'doesnotexist')) is True

    def test_create_program_info_file(self, monkeypatch, tmp_path):
        file_path = tmp_path / 'file.pro'
        file_path.write_text('dummy')
        class FakeCitation:
            @staticmethod
            def create_from_file(fp):
                class Fake:
                    def write(self, outfp):
                        FakeCitation.called = (fp, outfp)
                return Fake()
        monkeypatch.setattr('get_program_info.Citation_Information', FakeCitation)
        monkeypatch.setattr('get_program_info.PROGRAM_INFO_FILE', 'program-info.txt')
        create_program_info_file(str(file_path))
        assert hasattr(FakeCitation, 'called')
        assert FakeCitation.called[0] == str(file_path)
        assert FakeCitation.called[1].endswith('program-info.txt')

    def test_download_proposal_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr('get_program_info.get_formatted_proposal_id', lambda pid: '12345')
        monkeypatch.setattr('get_program_info.DOCUMENT_EXT', ('pro', 'apt'))
        monkeypatch.setattr('get_program_info.DOCUMENT_EXT_FOR_CITATION_INFO', ('pro',))
        monkeypatch.setattr('get_program_info.fs', types.SimpleNamespace(path=os.path))
        monkeypatch.setattr('get_program_info.is_proposal_file_retrieved', lambda pid, url, fp, logger=None: True)
        monkeypatch.setattr('get_program_info.create_program_info_file', lambda fp: None)
        logger = mock.Mock()
        logger.open = mock.Mock()
        logger.info = mock.Mock()
        res = download_proposal_files(12345, str(tmp_path), logger=logger)
        assert res == {'12345.pro', '12345.apt'}
