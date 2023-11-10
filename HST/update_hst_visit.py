##########################################################################################
# update_hst_visit/__init__.py
#
# update_hst_visit is the main function called in update_hst_visit pipeline task script.
# It will do these actions:
#
# - Queue retrieve_hst_visit and wait for it to complete.
# - Queue label_hst_products and wait for it to complete.
# - Queue prepare_browse_products and wait for it to complete.
##########################################################################################

import pdslogger

from queue_manager import queue_next_task
from queue_manager.task_queue_db import remove_a_prog_id_task_queue

def update_hst_visit(proposal_id, visit, logger=None):
    """Queue retrieve_hst_visit for the given visit and wait for it to complete.
    Queue label_hst_products for the given visit and wait for it to complete.
    Queue task prepare_browse_products for the given visit and wait for it to complete.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Update hst visit with proposal id: {proposal_id} & visit: {visit}')
    try:
        _ = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    logger.info(f'Queue retrieve_hst_visit for {proposal_id} visit {visit}')
    p1 = queue_next_task(proposal_id, visit, 5, logger)
    if p1 is not None:
        p1.communicate()

    logger.info(f'Queue label_hst_products for {proposal_id} visit {visit}')
    p2 = queue_next_task(proposal_id, visit, 6, logger)
    if p2 is not None:
        p2.communicate()

    logger.info(f'Queue prepare_browse_products for {proposal_id} visit {visit}')
    p3 = queue_next_task(proposal_id, visit, 7, logger)
    if p3 is not None:
        p3.communicate()

    # Remove the task queue for the given proposal id & visit from db
    remove_a_prog_id_task_queue(proposal_id, visit)
