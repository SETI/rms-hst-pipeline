##########################################################################################
# update_hst_visit/__init__.py
##########################################################################################
import pdslogger

from queue_manager import queue_next_task
from queue_manager.task_queue_db import remove_a_prog_id_task_queue

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
        _ = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    print(f'===========Queue in retrieve-hst-visit, task: 5 for {proposal_id} visit {visit}===========')
    p1 = queue_next_task(proposal_id, visit, 5, logger)
    p1.wait()
    print(f'===========Queue in label-hst-products, task: 6 for {proposal_id} visit {visit}===========')
    p2 = queue_next_task(proposal_id, visit, 6, logger)
    p2.wait()
    print(f'===========Queue in prepare-browse-products, task: 6 for {proposal_id}===========')
    p3 = queue_next_task(proposal_id, visit, 7, logger)
    p3.wait()
    # Remove the task queue for the given proposal id & visit
    remove_a_prog_id_task_queue(proposal_id, visit)
