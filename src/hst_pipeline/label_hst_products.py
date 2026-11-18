##########################################################################################
# label_hst_products.py
#
# label_hst_visit is the main function called in label_hst_products pipeline task script.
# It labels one visit after its downloads have been filed into collection directories
# under staging: a PDS4 label for every FITS file, then a Product_Browse label for every
# MAST preview.
##########################################################################################

import os
import pdslogger

from hst_pipeline.hst_helper.fs_utils import get_program_dir_path
from hst_pipeline.label_browse_products import label_browse_collection_directory
from hst_pipeline.product_labels import label_hst_fits_directories

DATA_COLLECTION_PREFIXES = ('data_', 'miscellaneous_')
BROWSE_COLLECTION_PREFIX = 'browse_'


def get_visit_collection_dirs(proposal_id, visit, prefixes):
    """Return the visit directories of every staged collection with one of the prefixes.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.
        prefixes       tuple of collection name prefixes, e.g., ('data_',
                       'miscellaneous_').

    Returns:    a sorted list of <HST_STAGING>/hst_<nnnnn>/<collection>/visit_<ss> paths
                that exist.
    """
    staging_root = get_program_dir_path(proposal_id, None, 'staging')
    if not os.path.isdir(staging_root):
        return []

    dirs = []
    for name in sorted(os.listdir(staging_root)):
        if not name.startswith(prefixes):
            continue
        visit_dir = os.path.join(staging_root, name, f'visit_{visit}')
        if os.path.isdir(visit_dir):
            dirs.append(visit_dir)
    return dirs


def label_hst_visit(proposal_id, visit, *, match_pattern='', old_directories=[],
                    retrieval_date='', logger=None, reset_dates=True):
    """Label the filed products of one visit.

    The FITS files of every data_* and miscellaneous_* collection for the visit are
    labeled together, in one call, so that each label can refer to the visit's other
    files (e.g., its spt, asn, and trl files). Then every browse_* collection holding the
    visit's previews is labeled.

    Inputs:
        proposal_id        a proposal id.
        visit              two character visit.
        match_pattern      optional fnmatch pattern to filter the FITS files labeled.
        old_directories    directories containing previous versions of the FITS files.
        retrieval_date     MAST retrieval date in yyyy-mm-dd format, or '' to infer it.
        logger             pdslogger to use; None for default EasyLogger.
        reset_dates        True to reset each FITS file's modification date.

    Returns:    a tuple (number of data directories labeled, number of browse labels).
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst products with proposal id: {proposal_id} & visit: {visit}')

    data_dirs = get_visit_collection_dirs(proposal_id, visit, DATA_COLLECTION_PREFIXES)
    if data_dirs:
        label_hst_fits_directories(data_dirs,
                                   match_pattern=match_pattern,
                                   old_directories=old_directories,
                                   retrieval_date=retrieval_date,
                                   logger=logger,
                                   reset_dates=reset_dates)
    else:
        logger.warn(f'No filed FITS files to label for visit {visit}')

    browse_labels = 0
    for visit_dir in get_visit_collection_dirs(proposal_id, visit,
                                               (BROWSE_COLLECTION_PREFIX,)):
        collection_dir = os.path.dirname(visit_dir)
        browse_labels += label_browse_collection_directory(collection_dir, proposal_id,
                                                           logger)

    return (len(data_dirs), browse_labels)
