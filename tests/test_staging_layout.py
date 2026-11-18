##########################################################################################
# tests/test_staging_layout.py
#
# Tests that each file is stored once in staging: prepare_browse_products moves the
# downloads of a visit from its mastDownload inbox into collection directories,
# label_hst_visit labels them there, the task queue runs the two in that order, and
# dedupe_staging removes the duplicate inboxes left by older versions of the pipeline.
##########################################################################################

import os

import pytest

from hst_pipeline import dedupe_staging as dedupe_module
from hst_pipeline import label_hst_products as label_module
from hst_pipeline import prepare_browse_products as prepare_module
from hst_pipeline import update_hst_visit as update_visit_module
from hst_pipeline.queue_manager.config import TASK_INFO

PROPOSAL_ID = 5167
VISIT = '01'


@pytest.fixture
def staging(tmp_path, monkeypatch):
    """Point every module's staging paths at tmp_path; return the program directory."""
    def fake_program_dir_path(proposal_id, visit=None, root_dir='staging',
                              testing=False):
        path = str(tmp_path / f'hst_{int(proposal_id):05d}')
        return path + (f'/visit_{visit}' if visit else '')

    for module in (prepare_module, label_module, dedupe_module):
        monkeypatch.setattr(module, 'get_program_dir_path', fake_program_dir_path)
    program_dir = tmp_path / f'hst_{PROPOSAL_ID:05d}'
    program_dir.mkdir()
    return program_dir


def _write(path, content=b'data'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(content)


def _inbox(program_dir, visit=VISIT):
    return program_dir / f'visit_{visit}' / 'mastDownload' / 'HST'


def _all_files(top):
    return sorted(os.path.relpath(os.path.join(root, name), top)
                  for (root, _, names) in os.walk(top) for name in names)


##########################################################################################
# prepare_browse_products: move, don't copy
##########################################################################################

def test_prepare_moves_each_file_into_its_collection(staging):
    inbox = _inbox(staging)
    _write(inbox / 'ld8x01010' / 'ld8x01010_x1dsum1.fits')
    _write(inbox / 'ld8x01010' / 'ld8x01010_x1dsum1.png')
    _write(inbox / 'ld8x01010' / 'ld8x01010_asn.fits')

    assert prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT) == 3

    assert _all_files(staging) == [
        'browse_cos_x1dsum/visit_01/ld8x01010_x1dsum1.png',
        'data_cos_x1dsum/visit_01/ld8x01010_x1dsum1.fits',
        'miscellaneous_cos_asn/visit_01/ld8x01010_asn.fits',
    ]
    # The emptied inbox is pruned.
    assert not (staging / 'visit_01' / 'mastDownload').exists()


def test_prepare_rerun_is_a_no_op(staging):
    _write(_inbox(staging) / 'ld8x01010' / 'ld8x01010_x1dsum.fits')
    prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT)
    before = _all_files(staging)

    assert prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT) == 0
    assert _all_files(staging) == before


def test_prepare_redownload_replaces_filed_copy(staging):
    _write(_inbox(staging) / 'ld8x01010' / 'ld8x01010_x1dsum.fits', b'old')
    prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT)

    _write(_inbox(staging) / 'ld8x01010' / 'ld8x01010_x1dsum.fits', b'new')
    prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT)

    filed = staging / 'data_cos_x1dsum' / 'visit_01' / 'ld8x01010_x1dsum.fits'
    assert filed.read_bytes() == b'new'
    assert _all_files(staging) == ['data_cos_x1dsum/visit_01/ld8x01010_x1dsum.fits']


def test_prepare_drops_identical_repeat_and_keeps_conflicting_one(staging):
    # MAST can deliver one file under two observations; the second is a repeat.
    inbox = _inbox(staging)
    _write(inbox / 'ld8x01010' / 'ld8x01010_x1dsum.fits', b'same')
    _write(inbox / 'ld8x01011' / 'ld8x01010_x1dsum.fits', b'same')
    _write(inbox / 'ld8x01010' / 'ld8x01010_x1d.fits', b'first')
    _write(inbox / 'ld8x01011' / 'ld8x01010_x1d.fits', b'second')

    prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT)

    assert (staging / 'data_cos_x1d' / 'visit_01' / 'ld8x01010_x1d.fits').read_bytes() \
        == b'first'
    # The conflicting repeat is not overwritten onto the filed copy; it stays in the inbox.
    assert _all_files(staging / 'visit_01') == [
        'mastDownload/HST/ld8x01011/ld8x01010_x1d.fits'
    ]


def test_prepare_leaves_unfiled_file_in_inbox(staging):
    _write(_inbox(staging) / 'j9xx01abq' / 'j9xx01abq_drz_large.jpg')
    _write(_inbox(staging) / 'j9xx01abq' / 'j9xx01abq_drz.jpg')

    assert prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT) == 1
    assert _all_files(staging) == [
        'browse_acs_drz/visit_01/j9xx01abq_drz.jpg',
        'visit_01/mastDownload/HST/j9xx01abq/j9xx01abq_drz_large.jpg',
    ]


def test_prepare_leaves_duplicate_named_file_in_inbox(staging):
    inbox = _inbox(staging)
    _write(inbox / 'odtf01010' / 'odtf01e5j_epc.fits')
    _write(inbox / 'odtf01020' / 'odtf01e5j_epc.fits-duplicate')

    assert prepare_module.prepare_browse_products(PROPOSAL_ID, VISIT) == 1
    assert _all_files(staging) == [
        'miscellaneous_stis_epc/visit_01/odtf01e5j_epc.fits',
        'visit_01/mastDownload/HST/odtf01020/odtf01e5j_epc.fits-duplicate',
    ]


@pytest.mark.parametrize('filename,expected', [
    ('ld8x01010_x1dsum3.fits', 'data_cos_x1dsum'),
    ('ld8x01010_x1dsum3.xml', 'data_cos_x1dsum'),
    ('ld8x01010_x1dsum3.png', 'browse_cos_x1dsum'),
    ('ld8x01010_asn.fits', 'miscellaneous_cos_asn'),
    ('u2no0401t_c0f.jpg', 'browse_wfpc2_c0m'),
    ('u2no0401t_drw.jpg', 'browse_wfpc2_drw'),
    ('j9xx01abq_drz_large.jpg', None),
    ('hst_05167_readme.txt', None),
    ('odtf01e5j_epc.fits-duplicate', None),     # renamed repeat from an old download
    ('ld8x01010_x1dsum.fits.gz', None),
])
def test_get_collection_dirname(filename, expected):
    assert prepare_module.get_collection_dirname(filename) == expected


##########################################################################################
# label_hst_visit: label the filed files
##########################################################################################

def test_label_hst_visit_labels_all_visit_collections_in_one_call(staging, monkeypatch):
    for path in ('data_cos_x1dsum/visit_01/ld8x01010_x1dsum.fits',
                 'data_cos_x1d/visit_01/ld8x01010_x1d.fits',
                 'miscellaneous_cos_spt/visit_01/ld8x01010_spt.fits',
                 'data_cos_x1d/visit_02/ld8x02010_x1d.fits',        # another visit
                 'browse_cos_x1dsum/visit_01/ld8x01010_x1dsum.png'):
        _write(staging / path)

    calls = []
    monkeypatch.setattr(label_module, 'label_hst_fits_directories',
                        lambda dirs, **kwargs: calls.append(dirs))

    assert label_module.label_hst_visit(PROPOSAL_ID, VISIT) == (3, 1)
    # One call, so each label can refer to the visit's other files (e.g., its spt file).
    assert calls == [[str(staging / 'data_cos_x1d' / 'visit_01'),
                      str(staging / 'data_cos_x1dsum' / 'visit_01'),
                      str(staging / 'miscellaneous_cos_spt' / 'visit_01')]]
    assert (staging / 'browse_cos_x1dsum' / 'visit_01' / 'ld8x01010_x1dsum.xml').exists()


def test_label_hst_visit_without_fits_files(staging, monkeypatch):
    calls = []
    monkeypatch.setattr(label_module, 'label_hst_fits_directories',
                        lambda dirs, **kwargs: calls.append(dirs))
    assert label_module.label_hst_visit(PROPOSAL_ID, VISIT) == (0, 0)
    assert calls == []


##########################################################################################
# Task order: retrieve, then file, then label
##########################################################################################

def test_task_order_files_before_labeling():
    order = {task: info[0] for (task, info) in TASK_INFO.items()}
    assert order['retrieve_visit'] < order['prep_browse_prod'] < order['label_prod']
    assert order['label_prod'] < order['finalize_bundle']


def test_update_hst_visit_queues_tasks_in_order(monkeypatch):
    queued = []
    monkeypatch.setattr(update_visit_module, 'queue_next_task',
                        lambda proposal_id, visit, task, logger: queued.append(task))
    monkeypatch.setattr(update_visit_module, 'is_a_task_done', lambda *args: True)
    monkeypatch.setattr(update_visit_module, 'remove_all_tasks_for_a_prog_id_and_visit',
                        lambda *args: None)

    update_visit_module.update_hst_visit(PROPOSAL_ID, VISIT)
    assert queued == ['retrieve_visit', 'prep_browse_prod', 'label_prod']


##########################################################################################
# dedupe_staging: remove the duplicate inboxes of older pipeline runs
##########################################################################################

def _old_layout(staging, visit='01', content=b'data'):
    """Lay out one visit as older pipelines left it: each file both in the inbox and in
    its collection."""
    for name, collection in (('ld8x01010_x1dsum.fits', 'data_cos_x1dsum'),
                             ('ld8x01010_x1dsum.xml', 'data_cos_x1dsum'),
                             ('ld8x01010_x1dsum.png', 'browse_cos_x1dsum')):
        _write(_inbox(staging, visit) / 'ld8x01010' / name, content)
        _write(staging / collection / f'visit_{visit}' / name, content)


def test_dedupe_reports_without_deleting(staging):
    _old_layout(staging)
    results = dedupe_module.dedupe_staging(PROPOSAL_ID)

    assert [(r.visit, r.redundant, r.problems, r.deleted) for r in results] == [
        ('01', 3, [], False)
    ]
    assert (staging / 'visit_01' / 'mastDownload').is_dir()


def test_dedupe_deletes_verified_inbox(staging):
    _old_layout(staging)
    _write(_inbox(staging) / 'j9xx01abq' / 'j9xx01abq_drz_large.jpg')   # never filed

    results = dedupe_module.dedupe_staging(PROPOSAL_ID, delete=True)

    assert [(r.redundant, len(r.unfiled), r.deleted) for r in results] == [(3, 1, True)]
    assert not (staging / 'visit_01' / 'mastDownload').exists()
    assert (staging / 'data_cos_x1dsum' / 'visit_01' / 'ld8x01010_x1dsum.fits').exists()


def test_dedupe_keeps_inbox_with_unverified_file(staging):
    _old_layout(staging)
    _write(staging / 'data_cos_x1dsum' / 'visit_01' / 'ld8x01010_x1dsum.fits', b'other')
    _write(_inbox(staging) / 'ld8x01010' / 'ld8x01010_x1d.fits')          # never filed

    results = dedupe_module.dedupe_staging(PROPOSAL_ID, delete=True)

    inbox = _inbox(staging) / 'ld8x01010'
    assert [(sorted(results[0].problems), results[0].deleted)] == [
        (sorted([str(inbox / 'ld8x01010_x1d.fits'),
                 str(inbox / 'ld8x01010_x1dsum.fits')]), False)
    ]
    assert len(_all_files(staging / 'visit_01')) == 4


def test_dedupe_handles_each_visit_separately(staging):
    _old_layout(staging, '01')
    _old_layout(staging, '02')
    _write(staging / 'data_cos_x1dsum' / 'visit_02' / 'ld8x01010_x1dsum.fits', b'other')

    results = dedupe_module.dedupe_staging(PROPOSAL_ID, delete=True)

    assert [(r.visit, r.deleted) for r in results] == [('01', True), ('02', False)]
    assert not (staging / 'visit_01' / 'mastDownload').exists()
    assert (staging / 'visit_02' / 'mastDownload').is_dir()


@pytest.mark.parametrize('content,deleted', [(b'data', True), (b'other', False)])
def test_dedupe_checks_duplicate_named_file_against_original(staging, content, deleted):
    _write(staging / 'miscellaneous_stis_epc' / 'visit_01' / 'odtf01e5j_epc.fits')
    _write(_inbox(staging) / 'odtf01020' / 'odtf01e5j_epc.fits-duplicate', content)

    results = dedupe_module.dedupe_staging(PROPOSAL_ID, delete=True)

    assert [(r.redundant, r.unfiled, r.deleted) for r in results] == [
        (1 if deleted else 0, [], deleted)
    ]
    assert (staging / 'visit_01' / 'mastDownload').exists() != deleted
