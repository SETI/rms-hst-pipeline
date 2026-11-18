##########################################################################################
# prepare_browse_products.py
#
# prepare_browse_products is the main function called in prepare_browse_products pipeline
# task script. It files the downloads of one visit: every file under the visit's
# mastDownload inbox (under staging) is MOVED into its collection directory (under
# staging), so each file exists only once in staging:
#   - browse_{inst_id}_{suffix}  (JPEG/PNG previews from MAST, names unchanged)
#   - data_{inst_id}_{suffix} or miscellaneous_{inst_id}_{suffix}
# Files that belong in no collection stay in the inbox, and a warning is logged. Labels
# are written afterward by the label_hst_products task, from the collection directories.
##########################################################################################

import filecmp
import os
import pdslogger

from hst_pipeline.product_labels.suffix_info import (ACCEPTED_SUFFIXES,
                                                     browse_collection_name,
                                                     collection_name,
                                                     is_accepted_browse_suffix)

from hst_pipeline.hst_helper import (INST_ID_DICT,
                                     BROWSE_PROD_EXT,
                                     MAST_DOWNLOAD_DIRNAME)
from hst_pipeline.hst_helper.fs_utils import (get_formatted_proposal_id,
                                              get_program_dir_path,
                                              get_instrument_id_from_fname,
                                              get_file_suffix)


def get_collection_dirname(filename):
    """Return the name of the collection that a downloaded file belongs in.

    Input:
        filename    the basename of a downloaded file, e.g., "ld8x01010_x1dsum1.fits",
                    its label "ld8x01010_x1dsum1.xml", or a preview "u2no0401t_c0f.jpg".

    Returns:    the collection name, e.g., "data_cos_x1dsum" or "browse_wfpc2_c0m"; None if
                the file belongs in no collection.
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
    mastDownload inbox into their collection directories under the staging directory.

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
    return moved


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
