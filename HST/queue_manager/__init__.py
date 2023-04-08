##########################################################################################
# queue_manager/__init__.py
##########################################################################################
import datetime
import os
import json
import pdslogger
import subprocess
import time

from sqlalchemy.exc import OperationalError

from product_labels.suffix_info import INSTRUMENT_NAMES
from hst_helper.fs_utils import get_program_dir_path
from hst_helper.general_utils import (date_time_to_date,
                                      get_citation_info,
                                      get_clean_target_text,
                                      get_collection_label_data,
                                      get_instrument_id_set)
from .task_queue_db import (add_a_prog_id_task_queue,
                            create_task_queue_table,
                            erase_all_task_queue,
                            remove_a_prog_id_task_queue,
                            update_a_prog_id_task_queue)

MAX_ALLOWED_TIME = 60
MAX_SUBPROCESS_CNT = 3
SUBPROCESS_LIST = []

# A task number dictionary keyed by task numbers. Each number represents a task, and the
# corresponding command template is stored as the value. {P} will be replaced by proposal
# id and {V} will be replaced by visit number.
TASK_NUM_TO_CMD_MAPPING = {
    0: 'python pipeline/pipeline_query_hst_moving_targets.py --prog-id {P}',
    1: 'python pipeline/pipeline_query_hst_products.py --prog-id {P}',
    3: 'python pipeline/pipeline_get_program_info.py --prog-id {P}',
    4: 'python pipeline/pipeline_retrieve_hst_visit.py --prog-id {P} --vi {V}',
    5: 'python pipeline/pipeline_label_hst_products.py --prog-id {P} --vi {V}',
    6: 'python pipeline/pipeline_prepare_browse_products.py --prog-id {P}',
    7: 'python pipeline/pipeline_finalize_hst_bundle.py --prog-id {P}',
}

def run_pipeline(proposal_ids, logger=None):
    """With a given list of proposal ids, run pipeline for each program id.

    Inputs:
        proposal_ids:    a list of proposal ids.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Run pipeline with proposal ids: {proposal_ids}')

    # Task queue template:
    # A dictionary keyed by proposal id, and store the corresponding tuple (task_num,
    # status). Status will be either 0 (waiting) or 1 (running) to represent the current
    # task status
    # {
    #     prog_id: (task_num, status)
    # }

    # Initialize queue manager, create an empty json file
    # json.dump({}, open("task_queue.json", "w"))

    # 1. init tq dict for prog id to start with task num 0

    try:
        create_task_queue_table()
    except OperationalError as e:
        if 'alreadt exists' in e.__repr__():
            pass
        else:
            logger.error('Failed to create task queue table!')

    tq_dict = {}
    for prog_id in proposal_ids:
        try:
            proposal_id = int(prog_id)
        except ValueError:
            # logger.exception(ValueError)
            logger.warn(f'Proposal id: {prog_id} is not valid.')
            pass
        #
        tq_dict[prog_id] = 0
        add_a_prog_id_task_queue(prog_id.zfill(5), 0, 0)
    # erase_all_task_queue()
    # update_a_prog_id_task_queue(prog_id,3,1)
    remove_a_prog_id_task_queue(prog_id)


        # Start pipeline for current prog_id


def queue_in_next_task(proposal_id, logger):
    """Queue in the next task for a given proposal id, wait for the open subprocess slot
    to execute the corresponding command.
    1. Read the task queue json to get the next task
    2. Queue in the next task
    3. Run the task (subprocess)

    Inputs:
        proposal_id:        the proposal if of the current task
        logger:             pdslogger to use; None for default EasyLogger.
    """
    return


def run_and_maybe_wait(args,  max_allowed_time, proposal_id, logger):
    """Run one subprocess, waiting as necessary for a slot to open up.

    Inputs:
        args:               the command of a subprocess to be executed.
        max_allowed_time:   the max time for a subprocess to be done before killing it.
        proposal_id:        the proposal if of the current task
        logger:             pdslogger to use; None for default EasyLogger.
    """
    # TODO: update the queue manager (json), waiting
    wait_for_subprocess()

    logger.debug("Spawning subprocess %s", str(args))
    # TODO: update the queue manager (json), running
    max_allowed_time = MAX_ALLOWED_TIME
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append((pid, time.time(), time.time()+max_allowed_time, proposal_id))

def wait_for_subprocess(all=False):
    """Wait for one (or all) subprocess slots to open up.

    Inputs:
        all:    a flag to determine if we are waiting for all subprocess slots to open up.
    """
    subprocess_count = MAX_SUBPROCESS_CNT

    if all:
       subprocess_count = 0

    while len(SUBPROCESS_LIST) > 0:
        for i in range(len(SUBPROCESS_LIST)):
            pid, proc_start_time, proc_max_time, prog_id = SUBPROCESS_LIST[i]

            if pid.poll() is not None:
                # The subprocess completed, make the slot available for next subprocess
                del SUBPROCESS_LIST[i]
                # TODO: update the queue manager (json), next step waiting
                break

        if len(SUBPROCESS_LIST) <= subprocess_count:
            # A slot opened up! Or all processes finished. Depending on what we're
            # waiting for.
            break
        time.sleep(1)
