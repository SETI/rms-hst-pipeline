##########################################################################################
# prepare_browse_products/__init__.py
##########################################################################################
import os
import pdslogger

from product_labels.suffix_info import (BROWSE_SUFFIX_INFO,
                                        ACCEPTED_BROWSE_SUFFIXES,
                                        INSTRUMENT_FROM_LETTER_CODE)

from hst_helper.fs_utils import get_program_dir_path

def prepare_browse_products(proposal_id, visit, logger=None, testing=False):
    """Retrieve all accepted files for a given proposal id & visit.

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

    files_dir = get_program_dir_path(proposal_id, visit, root_dir='staging')


    print("Print BROWSE_SUFFIX_INFO=================")
    print(files_dir)
    print(ACCEPTED_BROWSE_SUFFIXES)

    for root, dirs, files in os.walk(files_dir):
        print(root)
        print(dirs)
        print(files)
        # for file in files:
        #     print(os.path.join(root, file))

    return
