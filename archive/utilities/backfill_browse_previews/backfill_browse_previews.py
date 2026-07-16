#!/usr/bin/env python3
##########################################################################################
# backfill_browse_previews.py
#
# Standalone script to generate picmaker browse_generated_* OPUS-size JPG previews for
# existing HST_BUNDLES deliverables that predate the prepare_browse_products integration.
#
# Walks data_{inst}_{suffix}/ collections, and for each FITS whose (instrument, suffix)
# is in PICMAKER_BROWSE_SUFFIXES, writes four JPGs into:
#   browse_generated_{inst}_{suffix}/visit_{visit}/
#
# Does not create PDS4 product labels or collection inventory CSV files; re-run
# finalize_hst_bundle (or a label-only pass) if those need updating.
#
# Requires HST_BUNDLES, HST_PIPELINE, and HST_STAGING in the environment (or pass
# --bundles-root). Imports live pipeline helpers from the repo HST/ package.
#
# Syntax (from this directory):
#   python backfill_browse_previews.py [-h] [--proposal-id ID ...] [--bundles-root DIR]
#                                      [--dry-run] [--force] [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import sys
from pathlib import Path

# Repo layout: archive/utilities/backfill_browse_previews/ -> ../../../HST
_REPO_ROOT = Path(__file__).resolve().parents[3]
_HST_PACKAGE_DIR = _REPO_ROOT / 'HST'
if str(_HST_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_HST_PACKAGE_DIR))

import pdslogger

from generate_browse_previews import (generate_browse_previews,
                                      picmaker_browse_collection_name)
from hst_helper import HST_DIR
from hst_helper.fs_utils import (get_format_term,
                                 get_formatted_proposal_id,
                                 get_visit)
from product_labels.suffix_info import (PICMAKER_BROWSE_SUFFIXES,
                                        PICMAKER_OPUS_SIZE_SUFFIXES)


def parse_data_collection_name(dir_name):
    """Parse ``data_{inst}_{suffix}`` into ``(instrument_id, suffix)`` or None."""

    if not dir_name.startswith('data_'):
        return None
    parts = dir_name.split('_', 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1].upper(), parts[2]


def expected_preview_basenames(fits_name):
    """Return the four OPUS-size JPG basenames expected for one FITS product."""

    format_term = get_format_term(fits_name)
    return [f'{format_term}_{size}.jpg' for size in PICMAKER_OPUS_SIZE_SUFFIXES]


def previews_complete(out_dir, fits_name):
    """True if all four expected JPG previews already exist under ``out_dir``."""

    out_dir = Path(out_dir)
    return all((out_dir / name).is_file() for name in expected_preview_basenames(fits_name))


def visit_from_fits_path(fits_path):
    """Return the two-character visit from a FITS path under ``visit_XX/`` or the name."""

    parent = Path(fits_path).parent.name
    if parent.startswith('visit_') and len(parent) >= 8:
        return parent[6:8]
    return get_visit(get_format_term(Path(fits_path).name))


def iter_deliverable_dirs(bundles_root, proposal_ids=None):
    """Yield ``(proposal_id, deliverable_path)`` for bundles under ``bundles_root``.

    If ``proposal_ids`` is given, only those proposals are yielded (skipped when the
    deliverable directory is missing). Otherwise every ``hst_*`` program directory
    with an ``*-deliverable`` child is yielded.
    """

    bundles_root = Path(bundles_root)
    if proposal_ids:
        for proposal_id in proposal_ids:
            formatted = get_formatted_proposal_id(proposal_id)
            deliverable = (
                bundles_root / f'hst_{formatted}' / f'hst_{formatted}-deliverable'
            )
            if deliverable.is_dir():
                yield int(formatted), deliverable
            else:
                yield int(formatted), None
        return

    if not bundles_root.is_dir():
        return

    for entry in sorted(bundles_root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith('hst_'):
            continue
        suffix = entry.name[4:]
        if not suffix.isdigit():
            continue
        deliverable = entry / f'{entry.name}-deliverable'
        if deliverable.is_dir():
            yield int(suffix), deliverable


def iter_backfill_jobs(deliverable_path):
    """Yield ``(fits_path, out_dir, instrument_id, suffix, visit)`` jobs for one bundle.

    Only FITS under ``data_{inst}_{suffix}/`` whose pair is in
    ``PICMAKER_BROWSE_SUFFIXES`` are yielded.
    """

    deliverable_path = Path(deliverable_path)
    if not deliverable_path.is_dir():
        return

    for data_dir in sorted(deliverable_path.iterdir()):
        if not data_dir.is_dir():
            continue
        parsed = parse_data_collection_name(data_dir.name)
        if parsed is None:
            continue
        instrument_id, suffix = parsed
        allowed = PICMAKER_BROWSE_SUFFIXES.get(instrument_id, set())
        if suffix not in allowed:
            continue

        browse_col = picmaker_browse_collection_name(instrument_id, suffix)
        for root, _dirs, files in os.walk(data_dir):
            for name in sorted(files):
                if not name.lower().endswith('.fits'):
                    continue
                fits_path = Path(root) / name
                visit = visit_from_fits_path(fits_path)
                out_dir = deliverable_path / browse_col / f'visit_{visit}'
                yield fits_path, out_dir, instrument_id, suffix, visit


def backfill_deliverable(deliverable_path, logger=None, *, dry_run=False, force=False):
    """Generate missing browse_generated_* previews for one deliverable directory.

    Returns a dict with counts: ``processed``, ``skipped``, ``failed``.
    """

    logger = logger or pdslogger.EasyLogger()
    counts = {'processed': 0, 'skipped': 0, 'failed': 0}

    jobs = list(iter_backfill_jobs(deliverable_path))
    if not jobs:
        logger.info(f'No picmaker-eligible FITS under {deliverable_path}')
        return counts

    logger.info(f'Found {len(jobs)} picmaker-eligible FITS under {deliverable_path}')
    for fits_path, out_dir, instrument_id, suffix, visit in jobs:
        if not force and previews_complete(out_dir, fits_path.name):
            logger.info(
                f'Skip (previews exist): {fits_path.name} '
                f'({instrument_id}/{suffix} visit_{visit})'
            )
            counts['skipped'] += 1
            continue

        if dry_run:
            logger.info(
                f'[dry-run] Would generate previews for {fits_path} -> {out_dir}'
            )
            counts['processed'] += 1
            continue

        try:
            ok = generate_browse_previews(
                str(fits_path), str(out_dir), instrument_id, suffix, logger,
            )
            if ok:
                counts['processed'] += 1
            else:
                counts['failed'] += 1
        except Exception as exc:
            logger.exception(exc)
            logger.error(f'Failed to generate previews for {fits_path}: {exc}')
            counts['failed'] += 1

    return counts


def backfill_browse_previews(proposal_ids=None, bundles_root=None, logger=None,
                             *, dry_run=False, force=False):
    """Backfill browse_generated_* previews for one or more HST_BUNDLES deliverables.

    Inputs:
        proposal_ids    optional list of proposal ids; None scans all deliverables.
        bundles_root    root directory containing ``hst_<nnnnn>/`` trees; defaults to
                        ``HST_DIR['bundles']`` (``HST_BUNDLES``).
        logger          pdslogger to use; None for default EasyLogger.
        dry_run         if True, log work without calling picmaker.
        force           if True, regenerate even when all four JPGs already exist.

    Returns:    aggregate counts dict with ``processed``, ``skipped``, ``failed``,
                and ``missing_deliverables``.
    """

    logger = logger or pdslogger.EasyLogger()
    bundles_root = bundles_root or HST_DIR['bundles']
    totals = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'missing_deliverables': 0,
    }

    logger.info(
        f'Backfill browse_generated_* under {bundles_root}'
        + (' (dry-run)' if dry_run else '')
    )

    found_any = False
    for proposal_id, deliverable in iter_deliverable_dirs(bundles_root, proposal_ids):
        found_any = True
        if deliverable is None:
            formatted = get_formatted_proposal_id(proposal_id)
            logger.warning(
                f'Missing deliverable for proposal {formatted} under {bundles_root}'
            )
            totals['missing_deliverables'] += 1
            continue

        logger.info(f'Backfill proposal {get_formatted_proposal_id(proposal_id)}: '
                    f'{deliverable}')
        counts = backfill_deliverable(
            deliverable, logger, dry_run=dry_run, force=force,
        )
        for key in ('processed', 'skipped', 'failed'):
            totals[key] += counts[key]

    if proposal_ids is None and not found_any:
        logger.warning(f'No hst_*-deliverable directories found under {bundles_root}')

    logger.info(
        f'Backfill complete: processed={totals["processed"]}, '
        f'skipped={totals["skipped"]}, failed={totals["failed"]}, '
        f'missing_deliverables={totals["missing_deliverables"]}'
    )
    return totals


def parse_args(argv=None):
    """Parse and return command-line arguments."""

    parser = argparse.ArgumentParser(
        description="""backfill_browse_previews: Generate picmaker browse_generated_*
                    OPUS-size JPG previews for existing HST_BUNDLES deliverables.""")

    parser.add_argument('--proposal-id', type=str, action='append', default=None,
        dest='proposal_ids',
        help="""Proposal id to backfill. May be repeated. If omitted, every
             hst_*-deliverable under the bundles root is processed.""")

    parser.add_argument('--bundles-root', type=str, default='',
        help="""Root directory containing hst_<nnnnn>/ trees. Defaults to the
             HST_BUNDLES environment variable.""")

    parser.add_argument('--dry-run', action='store_true',
        help='Log eligible FITS and output paths without calling picmaker.')

    parser.add_argument('--force', action='store_true',
        help='Regenerate previews even when all four JPGs already exist.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and
             time appended. If not specified, the file is written under the pipeline
             logs directory.""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def setup_logger(args, argv=None):
    """Configure and open the backfill logger."""

    logger = pdslogger.PdsLogger('pds.hst.backfill-browse-previews')
    if not args.quiet:
        logger.add_handler(pdslogger.stdout_handler)

    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if args.log:
        if os.path.isdir(args.log):
            logpath = os.path.join(args.log, 'backfill-browse-previews-' + now + '.log')
        else:
            parts = os.path.splitext(args.log)
            logpath = parts[0] + '-' + now + parts[1]
    else:
        log_dir = os.path.join(HST_DIR['pipeline'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logpath = os.path.join(log_dir, f'backfill-browse-previews-{now}.log')

    logger.add_handler(pdslogger.file_handler(logpath))
    limits = {'info': -1, 'debug': -1, 'normal': -1}
    cmd_args = sys.argv[1:] if argv is None else argv
    logger.open('backfill-browse-previews ' + ' '.join(cmd_args), limits=limits)
    return logger


def main(argv=None):
    """Run the backfill_browse_previews CLI."""

    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        bundles_root = args.bundles_root or None
        totals = backfill_browse_previews(
            proposal_ids=args.proposal_ids,
            bundles_root=bundles_root,
            logger=logger,
            dry_run=args.dry_run,
            force=args.force,
        )
        if totals['failed'] or totals['missing_deliverables']:
            return 1
        return 0
    except Exception as exc:
        logger.exception(exc)
        raise
    finally:
        logger.close()


if __name__ == '__main__':
    sys.exit(main() or 0)
