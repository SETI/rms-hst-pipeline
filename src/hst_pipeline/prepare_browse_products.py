##########################################################################################
# prepare_browse_products.py
#
# prepare_browse_products is the main function called in prepare_browse_products pipeline
# task script. It files the downloads of one visit: every file under the visit's
# mastDownload inbox (under staging) is MOVED into its collection directory (under
# staging), so each file exists only once in staging:
#   - browse_{inst_id}_{suffix}  (MAST previews of non-image products, names unchanged)
#   - data_{inst_id}_{suffix} or miscellaneous_{inst_id}_{suffix}
# Files that belong in no collection stay in the inbox, and a warning is logged. Then,
# picmaker generates a preview of each of the visit's image products (as listed in
# PICMAKER_BROWSE_SUFFIXES) that lacks one, in browse_{inst_id}_{suffix}. Labels are
# written afterward by the label_hst_products task, from the collection directories.
##########################################################################################

import filecmp
import os
import pdslogger

from astropy.io import fits
from picmaker.options import get_parser as get_picmaker_parser
from picmaker.picmaker import picmaker

from hst_pipeline.product_labels.suffix_info import (ACCEPTED_SUFFIXES,
                                                     PICMAKER_BROWSE_OPTIONS,
                                                     PICMAKER_BROWSE_REQUIREMENTS,
                                                     PICMAKER_BROWSE_SUFFIXES,
                                                     HST_DQ_SUFFIXES,
                                                     browse_collection_name,
                                                     collection_name,
                                                     is_accepted_browse_suffix)

from hst_pipeline.hst_helper import (INST_ID_DICT,
                                     BROWSE_PROD_EXT,
                                     LABEL_SUFFIX,
                                     MAST_DOWNLOAD_DIRNAME)
from hst_pipeline.hst_helper.fs_utils import (get_formatted_proposal_id,
                                              get_program_dir_path,
                                              get_instrument_id_from_fname,
                                              get_file_suffix)

# Extensions of the downloaded data files and their labels. Any other file, e.g., a
# "*.fits-duplicate" left by an older download step, belongs in no collection.
DATA_EXTENSIONS = ('fits', LABEL_SUFFIX.lstrip('.'))

# Format of the browse products that picmaker generates.
PICMAKER_EXTENSION = 'jpg'


def get_collection_dirname(filename):
    """Return the name of the collection that a downloaded file belongs in.

    Input:
        filename    the basename of a downloaded file, e.g., "ld8x01010_x1dsum1.fits",
                    its label "ld8x01010_x1dsum1.lblx", or a preview "j9xx01abq_drz.jpg".

    Returns:    the collection name, e.g., "data_cos_x1dsum" or "browse_acs_drz"; None if
                the file belongs in no collection, including any file whose extension is
                not one of DATA_EXTENSIONS or BROWSE_PROD_EXT.
    """
    inst_id = get_instrument_id_from_fname(filename)
    if inst_id is None:
        return None

    suffix = get_file_suffix(filename)
    _, _, ext = filename.rpartition('.')
    if ext.lower() in BROWSE_PROD_EXT:
        if is_accepted_browse_suffix(suffix, inst_id):
            return browse_collection_name(suffix, inst_id)
        return None
    if ext.lower() not in DATA_EXTENSIONS:
        return None
    if suffix in ACCEPTED_SUFFIXES[inst_id]:
        return collection_name(suffix, inst_id)
    return None


def get_filed_path(proposal_id, visit, filename):
    """Return the path in staging where a downloaded file is filed.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.
        filename       the basename of a downloaded file.

    Returns:    <HST_STAGING>/hst_<nnnnn>/<collection>/visit_<ss>/<filename>, or None if
                the file belongs in no collection.
    """
    collection = get_collection_dirname(filename)
    if collection is None:
        return None
    staging_root = get_program_dir_path(proposal_id, None, 'staging')
    return os.path.join(staging_root, collection, f'visit_{visit}', filename)


def prepare_browse_products(proposal_id, visit, logger=None):
    """With a given proposal id & visit, move the downloaded files of the visit from the
    mastDownload inbox into their collection directories under the staging directory, then
    generate the visit's missing picmaker browse products.

    A file that replaces one already filed (e.g., after a re-download) overwrites it. If
    two downloaded files share a basename, an identical repeat is dropped; a repeat with
    different content stays in the inbox and an error is logged. Directories left empty
    in the inbox are removed.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.
        logger         pdslogger to use; None for default EasyLogger.

    Returns:    the number of files moved.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info('Prepare browse products with '
                f'proposal id: {proposal_id} & visit: {visit}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    visit_dir = get_program_dir_path(proposal_id, visit, root_dir='staging')
    inbox = os.path.join(visit_dir, MAST_DOWNLOAD_DIRNAME)

    moved = 0
    filed_this_run = set()
    for root, dirs, files in os.walk(inbox):
        dirs.sort()     # visit observation directories in a reproducible order
        for file in sorted(files):
            file_path = os.path.join(root, file)
            inst_id = get_instrument_id_from_fname(file)
            if inst_id is not None:
                INST_ID_DICT[formatted_proposal_id].add(inst_id)

            dest = get_filed_path(proposal_id, visit, file)
            if dest is None:
                logger.warn(f'File belongs in no collection; left in place: {file_path}')
                continue

            if dest in filed_this_run:
                if filecmp.cmp(file_path, dest, shallow=False):
                    logger.info(f'Identical repeat of a filed file, removed: {file_path}')
                    os.remove(file_path)
                else:
                    logger.error('Same basename, different content; left in place',
                                 file_path)
                continue

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(file_path, dest)
            filed_this_run.add(dest)
            moved += 1
            logger.info(f'Moved {file} to: {dest}')

    _remove_empty_dirs(inbox, logger)
    logger.info(f'Moved {moved} file(s) from {inbox}')

    generate_picmaker_browse_products(proposal_id, visit, logger)
    return moved


def generate_picmaker_browse_products(proposal_id, visit, logger=None):
    """Generate the picmaker browse products of one visit.

    For each instrument and suffix in PICMAKER_BROWSE_SUFFIXES, every FITS file of the
    visit in data_{inst_id}_{suffix} that has no preview yet, and whose primary header
    meets PICMAKER_BROWSE_REQUIREMENTS, gets one, made with PICMAKER_BROWSE_OPTIONS and
    named after it. A WFPC or WFPC2 image listed in HST_DQ_SUFFIXES is masked by the
    data quality file of the same visit's data quality collection, e.g.,
    data_wfpc2_c1m/visit_04/u2no0401t_c1m.fits. The preview is written in
    the visit directory of browse_{inst_id}_{suffix}, e.g.,
    browse_wfpc2_c0m/visit_04/u2no0401t_c0m.jpg for
    data_wfpc2_c0m/visit_04/u2no0401t_c0m.fits.
    An existing preview is kept. A FITS file picmaker cannot convert is logged as an error
    and skipped.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.
        logger         pdslogger to use; None for default EasyLogger.

    Returns:    the number of browse products generated.
    """
    logger = logger or pdslogger.EasyLogger()
    staging_root = get_program_dir_path(proposal_id, None, 'staging')

    generated = 0
    for (inst_id, suffixes) in PICMAKER_BROWSE_SUFFIXES.items():
        for suffix in suffixes:
            data_dir = os.path.join(staging_root, collection_name(suffix, inst_id),
                                    f'visit_{visit}')
            if not os.path.isdir(data_dir):
                continue
            browse_dir = os.path.join(staging_root,
                                      browse_collection_name(suffix, inst_id),
                                      f'visit_{visit}')
            options = PICMAKER_BROWSE_OPTIONS
            dq_suffix = HST_DQ_SUFFIXES.get(inst_id, {}).get(suffix)
            if dq_suffix:
                dq_dir = os.path.join(staging_root, collection_name(dq_suffix, inst_id),
                                      f'visit_{visit}')
                options = {**options, 'hst_dq': dq_dir}
            for file in sorted(os.listdir(data_dir)):
                stem, _, ext = file.rpartition('.')
                if ext.lower() != 'fits' or get_instrument_id_from_fname(file) != inst_id:
                    continue
                browse_path = os.path.join(browse_dir, f'{stem}.{PICMAKER_EXTENSION}')
                if os.path.exists(browse_path):
                    continue
                fits_path = os.path.join(data_dir, file)
                if not _meets_picmaker_requirements(fits_path, inst_id):
                    continue
                try:
                    make_picmaker_browse_product(fits_path, browse_dir, options)
                except Exception as e:
                    logger.error(f'picmaker failed on {fits_path}: {e}')
                    continue
                if not os.path.exists(browse_path):
                    logger.error(f'picmaker wrote no browse product for {fits_path}')
                    continue
                generated += 1
                logger.info(f'Generated browse product: {browse_path}')

    if generated:
        logger.info(f'Generated {generated} picmaker browse product(s) for visit {visit}')
    return generated


def _meets_picmaker_requirements(fits_path, inst_id):
    """True if the primary header of a FITS file has the values that
    PICMAKER_BROWSE_REQUIREMENTS lists for its instrument, e.g., OBSTYPE = "IMAGING" for
    STIS.

    Inputs:
        fits_path    path to the FITS file.
        inst_id      its instrument ID.

    Returns:    True if picmaker should generate the file's preview.
    """
    requirements = PICMAKER_BROWSE_REQUIREMENTS.get(inst_id)
    if not requirements:
        return True
    header = fits.getheader(fits_path)
    return all(header.get(key) == value for (key, value) in requirements.items())


def make_picmaker_browse_product(fits_path, browse_dir, options):
    """Run picmaker on one FITS file, writing its preview to browse_dir under the FITS
    file's name with the extension PICMAKER_EXTENSION.

    Inputs:
        fits_path     path to the FITS file.
        browse_dir    the directory to write the preview to; created if necessary.
        options       dict of picmaker options, e.g., {"mosaic": True, "footprint": 7}.
    """
    # Start from picmaker's own defaults, which picmaker() requires to be complete.
    picmaker_options = vars(get_picmaker_parser().parse_args([fits_path]))
    picmaker_options.update(options)
    picmaker_options.update(directory=browse_dir, extension=PICMAKER_EXTENSION,
                            replace='all')
    picmaker(**picmaker_options)


def _remove_empty_dirs(top, logger):
    """Remove empty directories under and including top, deepest first.

    Inputs:
        top       directory to prune.
        logger    pdslogger to use.
    """
    if not os.path.isdir(top):
        return
    for root, _, _ in os.walk(top, topdown=False):
        if not os.listdir(root):
            os.rmdir(root)
            logger.debug(f'Removed empty directory: {root}')
