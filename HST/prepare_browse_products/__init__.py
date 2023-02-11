##########################################################################################
# prepare_browse_products/__init__.py
##########################################################################################
import os
import pdslogger
import shutil

from product_labels.suffix_info import ACCEPTED_BROWSE_SUFFIXES

from hst_helper import INST_ID_DICT
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path,
                                 get_instrument_id_from_fname,
                                 get_file_suffix)

def prepare_browse_products(proposal_id, visit, logger=None):
    """With a given proposal id & visit, save browse products to browse_{inst_id}_{suffix}
    under staging dir.

    Inputs:
        proposal_id:    a proposal id.
        visit:          two character visit.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info('Prepare browse products with '
                + f'proposal id: {proposal_id} & visit: {visit}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Walk through all the downloaded files from MAST (with ACCEPTED_SUFFIXES)
    files_dir = get_program_dir_path(proposal_id, visit, root_dir='staging')
    for root, dirs, files in os.walk(files_dir):
        for file in files:
            fp = os.path.join(root, file)
            suffix = get_file_suffix(file)
            inst_id = get_instrument_id_from_fname(file)

            formatted_proposal_id = get_formatted_proposal_id(proposal_id)
            INST_ID_DICT[formatted_proposal_id].add(inst_id)

            # Copy & save browse products under browse_{inst_id}_{suffix}
            if inst_id is not None and suffix in ACCEPTED_BROWSE_SUFFIXES[inst_id]:
                browse_dir = get_program_dir_path(proposal_id, None, 'staging')
                browse_dir += f"/browse_{inst_id.lower()}_{suffix}/visit_{visit}/"
                os.makedirs(browse_dir, exist_ok=True)
                logger.info(f'Move browse products to: {browse_dir+file}')
                shutil.copy(fp, browse_dir+file)

    return
