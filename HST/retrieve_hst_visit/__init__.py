##########################################################################################
# retrieve_hst_visit/__init__.py
##########################################################################################
import os
import pdslogger

from hst_helper.query_utils import (download_files,
                                    get_filtered_products,
                                    query_mast_slice)
from hst_helper.fs_utils import get_program_dir_path

def retrieve_hst_visit(proposal_id, visit, logger=None, testing=False):
    """Retrieve all accepted files for a given proposal id & visit.

    Inputs:
        proposal_id:    a proposal id.
        visit:          two character visit.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Retrieve hst visit with proposal id: {proposal_id} & visit: {visit}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Query mast
    table = query_mast_slice(proposal_id=proposal_id, logger=logger)
    filtered_products = get_filtered_products(table, visit)
    files_dir = get_program_dir_path(proposal_id, visit, root_dir='staging')
    # Download all accepted files
    download_files(filtered_products, files_dir, logger, testing)

    return len(filtered_products)
