##########################################################################################
# update_hst_program/__init__.py
##########################################################################################
import pdslogger

from queue_manager import queue_next_task
from queue_manager.task_queue_db import remove_all_task_queue_for_a_prog_id

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

    logger.info(f'Queue get_program_info for {proposal_id}')
    p1 = queue_next_task(proposal_id, '', 3, logger)
    p1.wait()

    pid_li = []
    for vi in visit_li:
        logger.info(f'Queue update_hst_visit for {proposal_id} visit {vi}')
        pid = queue_next_task(proposal_id, vi, 4, logger)
        pid_li.append(pid)
    for p in pid_li:
        p.wait()

    logger.info(f'Queue finalize_hst_bundle for {proposal_id}')
    p2 =  queue_next_task(proposal_id, '', 8, logger)
    p2.wait()
    # Remove all task queue for the given proposal id
    remove_all_task_queue_for_a_prog_id(proposal_id)
    # TODO: Add finish notification
