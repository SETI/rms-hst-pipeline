##########################################################################################
# update_hst_program/__init__.py
##########################################################################################
import pdslogger

from queue_manager import queue_next_task
from queue_manager.task_queue_db import remove_all_task_queue_for_a_prog_id
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

    logger.info(f'Update hst program with proposal id: {proposal_id} '
                + f'& visit li: {visit_li}')
    try:
        _ = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    print(f'===========Queue in get_program_info, task: 3 for {proposal_id}===========')
    p1 = queue_next_task(proposal_id, '', 3, logger)
    p1.wait()
    pid_li = []
    for vi in visit_li:
       print(f'===========Queue in update_hst_visit, task: 4 for {proposal_id} & vi {vi}===========')
       pid = queue_next_task(proposal_id, vi, 4, logger)
       pid_li.append(pid)
    for p in pid_li:
        p.wait()
    print(f'===========Queue in finalize-hst-bundle, task: 8 for {proposal_id}===========')
    p2 =  queue_next_task(proposal_id, '', 8, logger)
    p2.wait()
    # Remove all task queue for the given proposal id
    remove_all_task_queue_for_a_prog_id(proposal_id)
    # TODO: Add finish notification
