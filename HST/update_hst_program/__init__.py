##########################################################################################
# update_hst_program/__init__.py
##########################################################################################
import pdslogger

from queue_manager import queue_next_task
from hst_helper.query_utils import (download_files,
                                    get_filtered_products,
                                    query_mast_slice)
from hst_helper.fs_utils import get_program_dir_path

def update_hst_program(proposal_id, visit_li, logger=None):
    """Overall task to create a new bundle or to manage the update of an existing bundle.

    Inputs:
        proposal_id:    a proposal id.
        visit_li:       a list of visits in which any files are new or changed.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Update hst program with proposal id: {proposal_id} & visit: {visit}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    print(f'===========Queue in get_program_info, task: 3 for {proposal_id}===========')
    queue_next_task(proposal_id, 'all', 3, logger)
    # for vi in visit_li:
