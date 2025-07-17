import os
import shutil
import pytest
from unittest import mock

import HST.organize_files as organize_files

COL_NAME_PREFIX = ['data_', 'miscellaneous_', 'browse_']
MAST_DOWNLOAD_DIRNAME = 'mastDownload'

@pytest.fixture
def fake_logger():
    class FakeLogger:
        def __init__(self):
            self.messages = []
        def info(self, msg):
            self.messages.append(msg)
    return FakeLogger()

@pytest.fixture
def patch_helpers(monkeypatch):
    monkeypatch.setattr(organize_files, 'COL_NAME_PREFIX', COL_NAME_PREFIX)
    monkeypatch.setattr(organize_files, 'MAST_DOWNLOAD_DIRNAME', MAST_DOWNLOAD_DIRNAME)

# organize_files_from_staging_to_bundles
@mock.patch('HST.organize_files.get_deliverable_path')
@mock.patch('HST.organize_files.get_program_dir_path')
@mock.patch('os.makedirs')
@mock.patch('shutil.copytree')
def test_organize_files_from_staging_to_bundles_basic(mock_copytree, mock_makedirs, mock_get_program_dir_path, mock_get_deliverable_path, fake_logger, patch_helpers, monkeypatch):
    proposal_id = '12345'
    staging_dir = '/fake/staging/hst_12345'
    deliverable_path = '/fake/bundles/hst_12345-deliverable'
    dirs = ['data_foo', 'miscellaneous_bar', 'browse_baz', 'other']
    mock_get_program_dir_path.return_value = staging_dir
    mock_get_deliverable_path.return_value = deliverable_path
    monkeypatch.setattr(os, 'listdir', lambda d: dirs)

    organize_files.organize_files_from_staging_to_bundles(proposal_id, fake_logger)

    # Only dirs starting with COL_NAME_PREFIX are copied
    expected = [
        mock.call(os.path.join(staging_dir, d), os.path.join(deliverable_path, d), dirs_exist_ok=True)
        for d in dirs if any(d.startswith(prefix) for prefix in COL_NAME_PREFIX)
    ]
    assert mock_copytree.call_args_list == expected
    # makedirs called for each
    assert mock_makedirs.call_count == 3
    # Logger messages
    assert any('Organize files for proposal id' in m for m in fake_logger.messages)
    assert any('Move data_foo' in m for m in fake_logger.messages)

# clean_up_staging_dir
@mock.patch('HST.organize_files.get_program_dir_path')
@mock.patch('os.path.isdir')
@mock.patch('os.listdir')
@mock.patch('shutil.rmtree')
def test_clean_up_staging_dir_basic(mock_rmtree, mock_listdir, mock_isdir, mock_get_program_dir_path, fake_logger, patch_helpers):
    proposal_id = '12345'
    staging_dir = '/fake/staging/hst_12345'
    dirs = ['mastDownload1', 'data_foo', 'other']
    mock_get_program_dir_path.return_value = staging_dir
    mock_isdir.return_value = True
    mock_listdir.return_value = dirs

    organize_files.clean_up_staging_dir(proposal_id, fake_logger)

    # mastDownload and data_foo should be removed
    expected = [
        mock.call(os.path.join(staging_dir, 'mastDownload1')),
        mock.call(os.path.join(staging_dir, 'data_foo')),
    ]
    assert mock_rmtree.call_args_list == expected
    # Logger messages
    assert any('Clean up staging directory' in m for m in fake_logger.messages)
    assert any('Remove mastDownload1' in m for m in fake_logger.messages)
    assert any('Remove data_foo' in m for m in fake_logger.messages)

@mock.patch('HST.organize_files.get_program_dir_path')
@mock.patch('os.path.isdir')
def test_clean_up_staging_dir_no_staging_dir(mock_isdir, mock_get_program_dir_path, fake_logger, patch_helpers):
    proposal_id = '12345'
    mock_get_program_dir_path.return_value = '/fake/staging/hst_12345'
    mock_isdir.return_value = False
    # Should not raise or call rmtree
    organize_files.clean_up_staging_dir(proposal_id, fake_logger)
    assert any('Clean up staging directory' in m for m in fake_logger.messages)

@mock.patch('HST.organize_files.get_program_dir_path')
@mock.patch('os.path.isdir')
@mock.patch('os.listdir')
@mock.patch('shutil.rmtree')
def test_clean_up_staging_dir_only_other(mock_rmtree, mock_listdir, mock_isdir, mock_get_program_dir_path, fake_logger, patch_helpers):
    proposal_id = '12345'
    staging_dir = '/fake/staging/hst_12345'
    dirs = ['other']
    mock_get_program_dir_path.return_value = staging_dir
    mock_isdir.return_value = True
    mock_listdir.return_value = dirs
    organize_files.clean_up_staging_dir(proposal_id, fake_logger)
    # No rmtree calls for dirs not matching prefix or mastDownload
    assert mock_rmtree.call_count == 0
