##########################################################################################
# update_hst_visit/__init__.py
##########################################################################################
import pdslogger

from queue_manager import queue_next_task
from hst_helper.query_utils import (download_files,
                                    get_filtered_products,
                                    query_mast_slice)
from hst_helper.fs_utils import get_program_dir_path

def update_hst_visit(proposal_id, visit, logger=None):
    """Queue retrieve_hst_visit for the given visit and wait for it to complete.
    Queue label_hst_products for the given visit and wait for it to complete.
    Queue task prepare_browse_products for the given visit and wait for it to complete.

    Inputs:
        proposal_id:    a proposal id.
        visit:          two character visit.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Update hst visit with proposal id: {proposal_id} & visit: {visit}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    print(f'===========Queue in retrieve-hst-visit, task: 5 for {proposal_id}===========')
    queue_next_task(proposal_id, visit, 5, logger)
