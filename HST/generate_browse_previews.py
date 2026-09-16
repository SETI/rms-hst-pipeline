##########################################################################################
# generate_browse_previews.py
#
# Generate OPUS-size browse JPG previews with picmaker for FITS products whose
# (instrument, suffix) pair is listed in PICMAKER_BROWSE_SUFFIXES. Output is
# written to browse_generated_{inst}_{suffix}/ collections.
##########################################################################################

import os
import re
import shutil
import tempfile
from pathlib import Path

import pdslogger
from picmaker.options import get_parser
from picmaker.picmaker import picmaker

from hst_helper.fs_utils import get_format_term
from product_labels.suffix_info import (PICMAKER_BROWSE_SUFFIXES,
                                        PICMAKER_OPUS_SIZE_SUFFIXES,
                                        use_mosaic)

_CONFIG_DIR = Path(__file__).resolve().parent / 'picmaker_configs'
_ACS_WFC3_CONFIG = _CONFIG_DIR / 'acs_wfc3_previews.txt'
_WFPC_WFPC2_CONFIG = _CONFIG_DIR / 'wfpc_wfpc2_previews.txt'
_FOC_CONFIG = _CONFIG_DIR / 'foc_previews.txt'
_NICMOS_CONFIG = _CONFIG_DIR / 'nicmos_previews.txt'
_STIS_CONFIG = _CONFIG_DIR / 'stis_previews.txt'
_COS_FGS_FOS_GHRS_HSP_CONFIG = _CONFIG_DIR / 'cos_fgs_fos_ghrs_hsp_previews.txt'

# Group instruments by shared preview behavior (stretch / tint / product era).
_ACS_WFC3_INSTRUMENTS = frozenset({'ACS', 'WFC3'})
_WFPC_WFPC2_INSTRUMENTS = frozenset({'WFPC', 'WFPC2'})
_COS_FGS_FOS_GHRS_HSP_INSTRUMENTS = frozenset({'COS', 'FGS', 'FOS', 'GHRS', 'HSP'})


def picmaker_browse_collection_name(instrument_id, suffix):
    """Return the staging collection name for picmaker-generated browse products.

    Inputs:
        instrument_id   HST instrument id (e.g. ACS, WFC3).
        suffix          FITS suffix (e.g. drz, raw).
    """

    return f'browse_generated_{instrument_id.lower()}_{suffix}'


def expected_opus_jpg_names(fits_path):
    """Return the four OPUS-size JPG basenames expected for one FITS product."""

    format_term = get_format_term(Path(fits_path).name)
    return [f'{format_term}_{size}.jpg' for size in PICMAKER_OPUS_SIZE_SUFFIXES]


def _get_picmaker_recipe(instrument_id, suffix):
    """Return (versions_path, extra_args) for an instrument and suffix.

    Suffix-uniform stretch, layout, tint, and trim flags live in the versions
    config file (sized tiers include ``--tint`` where appropriate; ``_full``
    does not). ``extra_args`` carries only suffix-varying options (``--mosaic``).

    Inputs:
        instrument_id   HST instrument id (e.g. ACS, WFC3).
        suffix          FITS suffix (e.g. drz, raw).
    """

    if instrument_id in _ACS_WFC3_INSTRUMENTS:
        config_path = _ACS_WFC3_CONFIG
    elif instrument_id in _WFPC_WFPC2_INSTRUMENTS:
        config_path = _WFPC_WFPC2_CONFIG
    elif instrument_id == 'FOC':
        config_path = _FOC_CONFIG
    elif instrument_id == 'NICMOS':
        config_path = _NICMOS_CONFIG
    elif instrument_id == 'STIS':
        config_path = _STIS_CONFIG
    elif instrument_id in _COS_FGS_FOS_GHRS_HSP_INSTRUMENTS:
        config_path = _COS_FGS_FOS_GHRS_HSP_CONFIG
    else:
        raise ValueError(
            f'No picmaker recipe for instrument {instrument_id!r} and suffix {suffix!r}'
        )

    extra_args = []
    if use_mosaic(instrument_id, suffix):
        extra_args = ['--mosaic']

    return config_path, extra_args


def _versions_file_for_suffix(versions_path, suffix):
    """Write a temporary versions file with ``--strip`` set for ``suffix``.

    Inputs:
        versions_path   picmaker versions file defining thumb/small/med/full.
        suffix          FITS suffix used to parameterize ``--strip`` in the versions file.
    """

    content = versions_path.read_text(encoding='utf-8')
    content = re.sub(r'--strip\s+_\S+', f'--strip _{suffix}', content)
    handle = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, encoding='utf-8',
    )
    handle.write(content)
    handle.close()
    return handle.name


def generate_hst_previews(fits_path, out_dir, versions_path, suffix, extra_args=()):
    """Run picmaker on one FITS file and write JPG previews to ``out_dir``.

    Inputs:
        fits_path       path to the input FITS file.
        out_dir         output directory for the generated JPGs.
        versions_path   picmaker versions file defining thumb/small/med/full.
        suffix          FITS suffix used to parameterize ``--strip`` in the versions file.
        extra_args      suffix-varying picmaker CLI tokens (e.g. --mosaic, --trim).
    """

    fits_path = Path(fits_path)
    out_dir = Path(out_dir)
    versions_path = Path(versions_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions_file = _versions_file_for_suffix(versions_path, suffix)
    try:
        args = [
            str(fits_path),
            '--directory', str(out_dir),
            '--extension', 'jpg',
            *extra_args,
            '--versions', versions_file,
        ]
        picmaker(**vars(get_parser().parse_args(args)))
    finally:
        os.unlink(versions_file)


def generate_browse_previews(fits_path, out_dir, instrument_id, suffix, logger=None):
    """Generate picmaker browse previews for one FITS product.

    Runs picmaker without ``--proceed``. On any picmaker failure, or if fewer than
    all four OPUS-size JPGs are produced, logs a warning and returns False without
    creating ``out_dir``. Only creates ``browse_generated_*`` paths when all four
    JPGs are ready to install.

    Inputs:
        fits_path       path to the FITS file in a data_{inst}_{suffix} collection.
        out_dir         browse_generated_{inst}_{suffix}/visit_{visit}/ directory.
        instrument_id   HST instrument id (e.g. ACS, WFC3).
        suffix          FITS suffix (e.g. drz, raw).
        logger          pdslogger to use; None for default EasyLogger.

    Returns:
        True if all four JPGs were written under ``out_dir``; False if generation
        was skipped after a warning.
    """

    logger = logger or pdslogger.EasyLogger()
    allowed_suffixes = PICMAKER_BROWSE_SUFFIXES.get(instrument_id)
    if not allowed_suffixes or suffix not in allowed_suffixes:
        raise ValueError(
            f'Picmaker browse generation is not configured for '
            f'{instrument_id!r} suffix {suffix!r}'
        )

    versions_path, extra_args = _get_picmaker_recipe(instrument_id, suffix)
    logger.info(
        f'Generate picmaker browse previews for {fits_path} '
        f'({instrument_id}/{suffix}) -> {out_dir}'
    )

    expected = expected_opus_jpg_names(fits_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            generate_hst_previews(
                fits_path, tmp_dir, versions_path, suffix, extra_args=extra_args,
            )
        except Exception as exc:
            logger.warning(
                f'Cannot generate all picmaker browse JPGs for {fits_path} '
                f'({instrument_id}/{suffix}): {exc}'
            )
            return False

        # This code block is for the case when picmaker doesn't raise an error and generate
        # any image.
        missing = [name for name in expected if not (tmp_dir / name).is_file()]
        if missing:
            found = sorted(path.name for path in tmp_dir.glob('*.jpg'))
            logger.warning(
                f'Cannot generate all picmaker browse JPGs for {fits_path} '
                f'({instrument_id}/{suffix}): missing {missing}; found {found}'
            )
            return False

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in expected:
            shutil.move(str(tmp_dir / name), str(out_dir / name))

    return True
