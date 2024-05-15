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

import time
import pdslogger

from queue_manager import queue_next_task
from queue_manager.task_queue_db import (is_a_task_done,
                                         remove_all_tasks_for_a_prog_id_and_visit)

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
    queue_next_task(proposal_id, visit, 'retrieve_visit', logger)

    while not is_a_task_done(proposal_id, visit, 'retrieve_visit'):
        time.sleep(60)
    logger.info(f'Retrieve hst visit for {proposal_id} visit {visit} has completed!')

    logger.info(f'Queue label_hst_products for {proposal_id} visit {visit}')
    queue_next_task(proposal_id, visit, 'label_prod', logger)
    while not is_a_task_done(proposal_id, visit, 'label_prod'):
        time.sleep(1)
    logger.info(f'Label hst products for {proposal_id} visit {visit} has completed!')

    logger.info(f'Queue prepare_browse_products for {proposal_id} visit {visit}')
    queue_next_task(proposal_id, visit, 'prep_browse_prod', logger)
    while not is_a_task_done(proposal_id, visit, 'prep_browse_prod'):
        time.sleep(1)
    logger.info(f'Prepare browse products for {proposal_id} visit {visit} has completed!')

    # Remove the task queue for the given proposal id & visit from db
    remove_all_tasks_for_a_prog_id_and_visit(proposal_id, visit)
