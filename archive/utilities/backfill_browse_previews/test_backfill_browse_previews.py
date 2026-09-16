##########################################################################################
# tests/test_backfill_browse_previews.py
##########################################################################################

from pathlib import Path
from unittest import mock

import pytest

import backfill_browse_previews as backfill


@pytest.fixture
def fake_logger():
    class FakeLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg):
            self.messages.append(('info', msg))

        def warning(self, msg):
            self.messages.append(('warning', msg))

        def error(self, msg):
            self.messages.append(('error', msg))

        def exception(self, exc):
            self.messages.append(('exception', str(exc)))

    return FakeLogger()


def test_parse_data_collection_name():
    assert backfill.parse_data_collection_name('data_wfc3_drz') == ('WFC3', 'drz')
    assert backfill.parse_data_collection_name('data_wfpc2_c0f') == ('WFPC2', 'c0f')
    assert backfill.parse_data_collection_name('browse_wfc3_drz') is None
    assert backfill.parse_data_collection_name('data_') is None


def test_expected_preview_basenames():
    assert backfill.expected_preview_basenames('iear01iaq_drz.fits') == [
        'iear01iaq_thumb.jpg',
        'iear01iaq_small.jpg',
        'iear01iaq_med.jpg',
        'iear01iaq_full.jpg',
    ]


def test_visit_from_fits_path(tmp_path):
    fits = tmp_path / 'data_wfc3_drz' / 'visit_01' / 'iear01iaq_drz.fits'
    fits.parent.mkdir(parents=True)
    fits.write_bytes(b'')
    assert backfill.visit_from_fits_path(fits) == '01'

    loose = tmp_path / 'iear02iaq_drz.fits'
    loose.write_bytes(b'')
    assert backfill.visit_from_fits_path(loose) == '02'


def _make_deliverable(root, proposal='16167'):
    deliverable = root / f'hst_{proposal}' / f'hst_{proposal}-deliverable'
    data_dir = deliverable / 'data_wfc3_drz' / 'visit_01'
    data_dir.mkdir(parents=True)
    fits = data_dir / 'iear01iaq_drz.fits'
    fits.write_bytes(b'fits')
    # Non-allowlisted suffix should be ignored
    other = deliverable / 'data_wfc3_flt' / 'visit_01'
    other.mkdir(parents=True)
    (other / 'iear01iaq_flt.fits').write_bytes(b'fits')
    return deliverable, fits


def test_iter_backfill_jobs(tmp_path):
    deliverable, fits = _make_deliverable(tmp_path)
    jobs = list(backfill.iter_backfill_jobs(deliverable))
    assert len(jobs) == 1
    fits_path, out_dir, instrument_id, suffix, visit = jobs[0]
    assert fits_path == fits
    assert instrument_id == 'WFC3'
    assert suffix == 'drz'
    assert visit == '01'
    assert out_dir == deliverable / 'browse_generated_wfc3_drz' / 'visit_01'


def test_iter_deliverable_dirs(tmp_path):
    deliverable, _ = _make_deliverable(tmp_path, proposal='16167')
    _make_deliverable(tmp_path, proposal='14930')

    found = list(backfill.iter_deliverable_dirs(tmp_path))
    assert [pid for pid, _ in found] == [14930, 16167]

    only = list(backfill.iter_deliverable_dirs(tmp_path, proposal_ids=['16167']))
    assert len(only) == 1
    assert only[0][0] == 16167
    assert only[0][1] == deliverable

    missing = list(backfill.iter_deliverable_dirs(tmp_path, proposal_ids=['99999']))
    assert missing == [(99999, None)]


def test_backfill_deliverable_dry_run_and_skip(tmp_path, fake_logger):
    deliverable, fits = _make_deliverable(tmp_path)
    out_dir = deliverable / 'browse_generated_wfc3_drz' / 'visit_01'

    with mock.patch.object(backfill, 'generate_browse_previews') as gen:
        counts = backfill.backfill_deliverable(
            deliverable, fake_logger, dry_run=True,
        )
        assert counts == {'processed': 1, 'skipped': 0, 'failed': 0}
        gen.assert_not_called()

    out_dir.mkdir(parents=True)
    for name in backfill.expected_preview_basenames(fits.name):
        (out_dir / name).write_bytes(b'jpg')

    with mock.patch.object(backfill, 'generate_browse_previews') as gen:
        counts = backfill.backfill_deliverable(deliverable, fake_logger)
        assert counts['skipped'] == 1
        assert counts['processed'] == 0
        gen.assert_not_called()

        counts = backfill.backfill_deliverable(
            deliverable, fake_logger, force=True,
        )
        assert counts['processed'] == 1
        assert counts['skipped'] == 0
        gen.assert_called_once()


def test_backfill_browse_previews_aggregates(tmp_path, fake_logger):
    _make_deliverable(tmp_path, proposal='16167')

    with mock.patch.object(backfill, 'generate_browse_previews'):
        totals = backfill.backfill_browse_previews(
            proposal_ids=['16167'],
            bundles_root=tmp_path,
            logger=fake_logger,
            dry_run=True,
        )
    assert totals['processed'] == 1
    assert totals['failed'] == 0
    assert totals['missing_deliverables'] == 0

    totals = backfill.backfill_browse_previews(
        proposal_ids=['99999'],
        bundles_root=tmp_path,
        logger=fake_logger,
        dry_run=True,
    )
    assert totals['missing_deliverables'] == 1
