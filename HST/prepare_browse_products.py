##########################################################################################
# prepare_browse_products.py
#
# prepare_browse_products is the main function called in prepare_browse_products pipeline
# task script. It will move data files from downloaded directories (under staging) to
# newly structured directories (under staging):
#   - browse_{inst_id}_{suffix}
#   - data_{inst_id}_{suffix}
##########################################################################################

import os
import pdslogger
import shutil

from product_labels.suffix_info import (ACCEPTED_BROWSE_SUFFIXES,
                                        ACCEPTED_SUFFIXES,
                                        collection_name)

from hst_helper import (INST_ID_DICT,
                        BROWSE_PROD_EXT)
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path,
                                 get_instrument_id_from_fname,
                                 get_file_suffix)

def prepare_browse_products(proposal_id, visit, logger=None):
    """With a given proposal id & visit, save browse products to browse_{inst_id}_{suffix}
    under staging dir.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info('Prepare browse products with '
                f'proposal id: {proposal_id} & visit: {visit}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Walk through all the downloaded files from MAST (with ACCEPTED_SUFFIXES)
    files_dir = get_program_dir_path(proposal_id, visit, root_dir='staging')
    for root, dirs, files in os.walk(files_dir):
        for file in files:
            file_path = os.path.join(root, file)
            suffix = get_file_suffix(file)
            inst_id = get_instrument_id_from_fname(file)

            formatted_proposal_id = get_formatted_proposal_id(proposal_id)
            if inst_id is not None:
                INST_ID_DICT[formatted_proposal_id].add(inst_id)

            _, _, file_ext = file.rpartition('.')
            # Copy & save browse products under browse_{inst_id}_{suffix} and rest of
            # products under data_{inst_id}_{suffix}
            # TODO: maybe use move instead of copy and remove empty folder (?)
            if inst_id is not None:
                prod_dir = get_program_dir_path(proposal_id, None, 'staging')
                col_name = collection_name(suffix, inst_id)
                if (suffix in ACCEPTED_BROWSE_SUFFIXES[inst_id] and
                    file_ext in BROWSE_PROD_EXT):
                    prod_dir += f'/browse_{inst_id.lower()}_{suffix}/visit_{visit}/'
                    logger.info(f'Move browse products to: {prod_dir + file}')
                elif suffix in ACCEPTED_SUFFIXES[inst_id]:
                    prod_dir += f'/{col_name}/visit_{visit}/'
                    logger.info(f'Move data products to: {prod_dir + file}')

                # Copy files to newly structured directories
                os.makedirs(prod_dir, exist_ok=True)
                shutil.copy(file_path, prod_dir + file)
                # shutil.move(file_path, prod_dir+file)
