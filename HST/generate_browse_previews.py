##########################################################################################
# generate_browse_previews.py
#
# Generate OPUS-size browse JPG previews with picmaker for FITS products whose
# (instrument, suffix) pair is listed in PICMAKER_BROWSE_SUFFIXES. Output is
# written to browse_generated_{inst}_{suffix}/ collections.
##########################################################################################

import os
import re
import tempfile
from pathlib import Path

import pdslogger
from picmaker.options import get_parser
from picmaker.picmaker import picmaker

from product_labels.suffix_info import (PICMAKER_BROWSE_SUFFIXES,
                                        use_mosaic)

_CONFIG_DIR = Path(__file__).resolve().parent / 'picmaker_configs'
_IMAGING_CONFIG = _CONFIG_DIR / 'acs_foc_wfc_wfc3_previews.txt'
_NICMOS_CONFIG = _CONFIG_DIR / 'nicmos_previews.txt'
_WFPC2_CONFIG = _CONFIG_DIR / 'wfpc2_previews.txt'
_SPECTRAL_CONFIG = _CONFIG_DIR / 'stis_cos_fgs_fos_ghrs_hsp_previews.txt'

_IMAGING_INSTRUMENTS = frozenset({'ACS', 'WFC3', 'FOC', 'WFPC'})
_SPECTRAL_INSTRUMENTS = frozenset({'STIS', 'COS', 'FGS', 'FOS', 'GHRS', 'HSP'})


def picmaker_browse_collection_name(instrument_id, suffix):
    """Return the staging collection name for picmaker-generated browse products."""

    return f'browse_generated_{instrument_id.lower()}_{suffix}'


def _get_picmaker_recipe(instrument_id, suffix):
    """Return (versions_path, extra_args, tint_sized) for an instrument and suffix.

    Suffix-uniform stretch and layout flags live in the versions config file.
    ``extra_args`` carries only suffix-varying options (``--mosaic``, ``--trim``,
    ``--percentiles`` where they differ by suffix). WFPC2 filter tint is injected
    into sized version lines at runtime because ``d0f`` must not be tinted.
    """

    tint_sized = False

    if instrument_id in _IMAGING_INSTRUMENTS:
        config_path = _IMAGING_CONFIG
        extra_args = []
    elif instrument_id == 'NICMOS':
        config_path = _NICMOS_CONFIG
        extra_args = []
    elif instrument_id == 'WFPC2':
        config_path = _WFPC2_CONFIG
        extra_args = []
        if suffix in {'c0f', 'raw', 'd0f', 'drz'}:
            extra_args.extend(['--trim', '100'])
        tint_sized = True
    elif instrument_id in _SPECTRAL_INSTRUMENTS:
        config_path = _SPECTRAL_CONFIG
        extra_args = []
        if suffix in {'raw', 'drz'}:
            extra_args = ['--percentiles', '0.02', '99.98']
    else:
        raise ValueError(
            f'No picmaker recipe for instrument {instrument_id!r} and suffix {suffix!r}'
        )

    if use_mosaic(instrument_id, suffix):
        extra_args = ['--mosaic', *extra_args]

    return config_path, extra_args, tint_sized


def _versions_file_for_suffix(versions_path, suffix, *, tint_sized=False):
    """Write a temporary versions file with ``--strip`` set for ``suffix``.

    When ``tint_sized`` is True, add ``--tint`` to sized tiers only (lines with
    ``--frame``), leaving the ``_full`` version untinted.
    """

    content = versions_path.read_text(encoding='utf-8')
    content = re.sub(r'--strip\s+_\S+', f'--strip _{suffix}', content)
    if tint_sized:
        lines = []
        for line in content.splitlines():
            if '--frame' in line and '--tint' not in line:
                line = line.replace('--frame', '--tint --frame', 1)
            lines.append(line)
        content = '\n'.join(lines) + '\n'
    handle = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, encoding='utf-8',
    )
    handle.write(content)
    handle.close()
    return handle.name


def generate_hst_previews(fits_path, out_dir, versions_path, suffix, extra_args=(),
                          *, tint_sized=False):
    """Run picmaker on one FITS file and write JPG previews to ``out_dir``.

    Inputs:
        fits_path       path to the input FITS file.
        out_dir         output directory for the generated JPGs.
        versions_path   picmaker versions file defining thumb/small/med/full.
        suffix          FITS suffix used to parameterize ``--strip`` in the versions file.
        extra_args      suffix-varying picmaker CLI tokens (e.g. --mosaic, --trim).
        tint_sized      if True, add ``--tint`` to sized version lines only.
    """

    fits_path = Path(fits_path)
    out_dir = Path(out_dir)
    versions_path = Path(versions_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions_file = _versions_file_for_suffix(
        versions_path, suffix, tint_sized=tint_sized,
    )
    try:
        args = [
            str(fits_path),
            '--directory', str(out_dir),
            '--proceed',
            '--extension', 'jpg',
            *extra_args,
            '--versions', versions_file,
        ]
        picmaker(**vars(get_parser().parse_args(args)))
    finally:
        os.unlink(versions_file)


def generate_browse_previews(fits_path, out_dir, instrument_id, suffix, logger=None):
    """Generate picmaker browse previews for one FITS product.

    Inputs:
        fits_path       path to the FITS file in a data_{inst}_{suffix} collection.
        out_dir         browse_generated_{inst}_{suffix}/visit_{visit}/ directory.
        instrument_id   HST instrument id (e.g. ACS, WFC3).
        suffix          FITS suffix (e.g. drz, raw).
        logger          pdslogger to use; None for default EasyLogger.
    """

    logger = logger or pdslogger.EasyLogger()
    allowed_suffixes = PICMAKER_BROWSE_SUFFIXES.get(instrument_id)
    if not allowed_suffixes or suffix not in allowed_suffixes:
        raise ValueError(
            f'Picmaker browse generation is not configured for '
            f'{instrument_id!r} suffix {suffix!r}'
        )

    versions_path, extra_args, tint_sized = _get_picmaker_recipe(instrument_id, suffix)
    logger.info(
        f'Generate picmaker browse previews for {fits_path} '
        f'({instrument_id}/{suffix}) -> {out_dir}'
    )
    generate_hst_previews(
        fits_path, out_dir, versions_path, suffix, extra_args=extra_args,
        tint_sized=tint_sized,
    )
