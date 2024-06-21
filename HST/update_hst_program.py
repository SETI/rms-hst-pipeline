##########################################################################################
# update_hst_program/__init__.py
#
# update_hst_program is the main function called in update_hst_program pipeline task
# script. It will do these actions:
#
# - Queue get_program_info and wait for it to complete.
# - For each visit in the visit list, queue update_hst_visit and wait until all visits
#   have completed.
# - Queue finialize_hst_bundle and wait for it to complete.
##########################################################################################

import pdslogger
import time

from queue_manager import queue_next_task
from queue_manager.task_queue_db import (is_a_task_done,
                                         remove_all_tasks_for_a_prog_id)

def update_hst_program(proposal_id, visit_li, logger=None):
    """Overall task to create a new bundle or to manage the update of an existing bundle.

    Inputs:
        proposal_id    a proposal id.
        visit_li       a list of visits in which any files are new or changed.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Update hst program with proposal id: {proposal_id} '
                f'& visit li: {visit_li}')
    try:
        _ = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    logger.info(f'Queue get_program_info for {proposal_id}')
    queue_next_task(proposal_id, '', 'get_prog_info', logger)

    while not is_a_task_done(proposal_id, '', 'get_prog_info'):
        time.sleep(1)
    logger.info(f'Get program info for {proposal_id} has completed!')

    for vi in visit_li:
        logger.info(f'Queue update_hst_visit for {proposal_id} visit {vi}')
        queue_next_task(proposal_id, vi, 'update_visit', logger)

    for vi in visit_li:
        while not is_a_task_done(proposal_id, vi, 'update_visit'):
            time.sleep(1)
    logger.info(f'All visits for {proposal_id} have completed update_hst_visit')

    logger.info(f'Queue finalize_hst_bundle for {proposal_id}')
    queue_next_task(proposal_id, '', 'finalize_bundle', logger)

    while not is_a_task_done(proposal_id, '', 'finalize_bundle'):
        time.sleep(1)
    # Remove all task queue & subprocess for the given proposal id from db
    logger.info(f'Pipeline is done. Remove all tasks from db for {proposal_id}')
    remove_all_tasks_for_a_prog_id(proposal_id)

    logger.info(f'HST pipeline for {proposal_id} is done')
