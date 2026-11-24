##########################################################################################
# tests/test_visit_lists.py
#
# Tests for the state that decides what a rerun redoes: each visit's products.txt and
# trl_checksums.txt, which change only once the visit has been labeled, and the backed-up
# proposal files that set the document's version.
##########################################################################################

import os

import pytest

from hst_pipeline import get_program_info
from hst_pipeline import query_hst_products
from hst_pipeline.finalize_document import count_document_revisions
from hst_pipeline.hst_helper import HST_DIR, PENDING_SUFFIX, PRODUCTS_FILE
from hst_pipeline.hst_helper.fs_utils import (commit_visit_lists,
                                              discard_pending_visit_lists)


@pytest.fixture
def visit_dir(tmp_path, monkeypatch):
    """Point HST_PIPELINE at tmp_path; return the directory of visit 01 of 12245."""
    monkeypatch.setitem(HST_DIR, 'pipeline', str(tmp_path))
    path = tmp_path / 'hst_12245' / 'visit_01'
    path.mkdir(parents=True)
    return path


def _query(files):
    """Compare the visit's products.txt with these files, as hst-query-products does."""
    return query_hst_products.compare_files_txt('12245', {'01': list(files)}, '01',
                                                PRODUCTS_FILE)


def _lines(path):
    return path.read_text().splitlines() if path.exists() else None


def test_lists_change_only_once_the_visit_is_labeled(visit_dir):
    products = visit_dir / PRODUCTS_FILE
    pending = visit_dir / (PRODUCTS_FILE + PENDING_SUFFIX)

    # First query: the visit is new; its list waits until the visit is labeled
    assert _query(['b_raw.fits', 'a_raw.fits'])
    assert _lines(products) is None
    assert _lines(pending) == ['a_raw.fits', 'b_raw.fits']

    # Never retrieved or labeled (as for 12245 visit 01): still new on the next query
    assert _query(['a_raw.fits', 'b_raw.fits'])

    # Labeled: the list is committed, and the next query finds nothing to do
    assert commit_visit_lists('12245', '01') == [str(products)]
    assert _lines(products) == ['a_raw.fits', 'b_raw.fits']
    assert not pending.exists()
    assert not _query(['a_raw.fits', 'b_raw.fits'])
    assert commit_visit_lists('12245', '01') == []

    # MAST adds a file: the old list stays until the visit is labeled again
    assert _query(['a_raw.fits', 'b_raw.fits', 'c_raw.fits'])
    assert _lines(products) == ['a_raw.fits', 'b_raw.fits']
    commit_visit_lists('12245', '01')
    assert _lines(products) == ['a_raw.fits', 'b_raw.fits', 'c_raw.fits']
    (backup,) = os.listdir(visit_dir / 'backups')
    assert backup.startswith('products-') and backup.endswith('.txt')


def test_failed_retrieval_discards_the_pending_lists(visit_dir):
    assert _query(['a_raw.fits'])
    discard_pending_visit_lists('12245', '01')
    assert os.listdir(visit_dir) == []
    assert commit_visit_lists('12245', '01') == []
    assert _query(['a_raw.fits'])           # still new


def test_query_matching_the_list_removes_a_stale_pending_list(visit_dir):
    (visit_dir / PRODUCTS_FILE).write_text('a_raw.fits\n')
    (visit_dir / (PRODUCTS_FILE + PENDING_SUFFIX)).write_text('a_raw.fits\nb_raw.fits\n')
    assert not _query(['a_raw.fits'])       # MAST withdrew b_raw.fits again
    assert os.listdir(visit_dir) == [PRODUCTS_FILE]


##########################################################################################
# document version
##########################################################################################

PROPOSAL_FILES = [('10782', '10782.apt', 'x'), ('10782', '10782.pdf', 'x'),
                  ('10782', '10782.pro', 'x')]


def test_count_document_revisions(tmp_path):
    assert count_document_revisions(str(tmp_path / 'backups'), PROPOSAL_FILES) == 0

    backups = tmp_path / 'backups'
    backups.mkdir()
    for name in ['10782-2026-01-02T03-04-05.apt',    # one download replaced three
                 '10782-2026-01-02T03-04-05.pdf',    # files: one revision
                 '10782-2026-01-02T03-04-05.pro',
                 '10782-2026-03-04T05-06-07.pdf',    # a later download: another
                 '10782-2026-05-06T07-08-09.prop',   # not one of the program's files
                 '10782.apt',                        # no time
                 'bundle-20261118145550',            # a copy made by hand
                 'image-previews-20261118162417']:
        (backups / name).write_text('')
    assert count_document_revisions(str(backups), PROPOSAL_FILES) == 2


def test_one_download_shares_one_backup_time(tmp_path, monkeypatch):
    times = []
    monkeypatch.setattr(get_program_info, 'is_proposal_file_retrieved',
                        lambda pid, url, fp, logger=None, now=None: times.append(now))
    get_program_info.download_proposal_files(10782, str(tmp_path))
    assert len(times) >= 3 and len(set(times)) == 1
