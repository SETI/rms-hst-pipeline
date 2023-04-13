##########################################################################################
# queue_manager/__init__.py
##########################################################################################
import os
import pdslogger
import subprocess
import sys
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
                            get_next_task_to_be_run,
                            init_task_queue_table,
                            remove_a_prog_id_task_queue,
                            update_a_prog_id_task_status,
                            update_a_prog_id_task_queue)

PYTHON_EXE = sys.executable
# root of pds-hst-pipeline dir
HST_SOURCE_ROOT = os.environ["PDS_HST_PIPELINE"]
# max allowed suborprocess time in seconds
MAX_ALLOWED_TIME = 60 * 60
MAX_SUBPROCESS_CNT = 3
SUBPROCESS_LIST = []

# Task to script command mapping. {P} will be replaced by proposal
# id and {V} will be replaced by visit number.
TASK_NUM_TO_CMD_MAPPING = {
    0: 'HST/pipeline/pipeline_query_hst_moving_targets.py --prog-id {P}',
    1: 'HST/pipeline/pipeline_query_hst_products.py --prog-id {P}',
    2: 'HST/pipeline/pipeline_get_program_info.py --prog-id {P}',
    3: 'HST/pipeline/pipeline_retrieve_hst_visit.py --prog-id {P} --vi {V}',
    4: 'HST/pipeline/pipeline_label_hst_products.py --prog-id {P} --vi {V}',
    5: 'HST/pipeline/pipeline_prepare_browse_products.py --prog-id {P}',
    6: 'HST/pipeline/pipeline_finalize_hst_bundle.py --prog-id {P}',
}

# Task to prority mapping. The larger the number, the higher the priority.
TASK_NUM_TO_PRI_MAPPING = {
    0: 1,
    1: 1,
    2: 5,
    3: 4,
    4: 5,
    5: 5,
    6: 5,
}

def run_pipeline(proposal_ids, logger=None):
    """With a given list of proposal ids, run pipeline for each program id.

    Inputs:
        proposal_ids:    a list of proposal ids.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Run pipeline with proposal ids: {proposal_ids}')

    try:
        init_task_queue_table()
    except OperationalError as e:
        if 'alreadt exists' in e.__repr__():
            pass
        else:
            logger.error('Failed to create task queue table!')

    for prog_id in proposal_ids:
        try:
            proposal_id = int(prog_id)
        except ValueError:
            # logger.exception(ValueError)
            logger.warn(f'Proposal id: {prog_id} is not valid.')
            pass
        proposal_id = str(proposal_id)
        print(f'============Queue in the first task: 0 for {proposal_id}===========')
        queue_in_next_task(proposal_id, 0, logger)
    # add_a_prog_id_task_queue('07885'.zfill(5), 2, 3, 0)
    # add_a_prog_id_task_queue('09059'.zfill(5), 4, 5, 0)
    # add_a_prog_id_task_queue('12345'.zfill(5), 3, 6, 1)
    # erase_all_task_queue()
    # update_a_prog_id_task_queue(prog_id,3,1)
    # remove_a_prog_id_task_queue(prog_id)


        # Start pipeline for current prog_id


def queue_in_next_task(proposal_id, task_num, logger):
    """Queue in the next task for a given proposal id to database, and wait for the open
    subprocess slot to execute the corresponding command. Once there is an open slot,
    update the task queue status
    1. Update the next task for a given proposal id to database. (task status: 0)
    2. Wait for an open slot in subprocess list, once there is an open slot, update
    the task status to 1 in the database.
    3. Run the task command (spawn the subprocess)

    Inputs:
        proposal_id:    the proposal if of the current task.
        task_num:       a number represents the current task.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Queue in the next task for: {proposal_id}, task num: {task_num}')

    priority = TASK_NUM_TO_PRI_MAPPING[0]
    # args = [sys.executable, 'pipeline/pipeline_query_hst_moving_targets.py', '--prog-id', '7885']
    cmd_parts = (TASK_NUM_TO_CMD_MAPPING[task_num].replace('{P}', proposal_id)
                                                  .split(' '))
    program_path = os.path.join(HST_SOURCE_ROOT, cmd_parts[0])
    args = [sys.executable, program_path] + cmd_parts[1::]
    max_allowed_time = MAX_ALLOWED_TIME
    add_a_prog_id_task_queue(proposal_id.zfill(5), task_num, priority, 0)
    run_and_maybe_wait(args,  max_allowed_time, proposal_id, logger)



def run_and_maybe_wait(args,  max_allowed_time, proposal_id, logger):
    """Run one subprocess, waiting as necessary for a slot to open up.

    Inputs:
        args:               the command of a subprocess to be executed.
        max_allowed_time:   the max time for a subprocess to be done before killing it.
        proposal_id:        the proposal if of the current task
        logger:             pdslogger to use; None for default EasyLogger.
    """
    wait_for_subprocess()

    # query database and see if there is a higher priority job waiting to be run, if so
    # spawn that subprocess first
    task = get_next_task_to_be_run()
    if task.proposal_id != proposal_id.zfill(5):
        proposal_id = task.proposal_id
        # TODO: handle visits {V}
        cmd_parts = (TASK_NUM_TO_CMD_MAPPING[task.task_num].replace('{P}', proposal_id)
                                                           .split(' '))
        program_path = os.path.join(HST_SOURCE_ROOT, cmd_parts[0])
        args = [sys.executable, program_path] + cmd_parts[1::]
        run_and_maybe_wait(args,  max_allowed_time, proposal_id, logger)
    else:
        # Update the task status to running (1)
        update_a_prog_id_task_status(proposal_id, 1)
        logger.debug("Spawning subprocess %s", str(args))
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

    cur_time = time.time()
    while len(SUBPROCESS_LIST) > 0:
        for i in range(len(SUBPROCESS_LIST)):
            pid, proc_start_time, proc_max_time, prog_id = SUBPROCESS_LIST[i]

            if pid.poll() is not None:
                # The subprocess completed, make the slot available for next subprocess
                del SUBPROCESS_LIST[i]
                # TODO: update the queue manager (json), next step waiting
                break

            if cur_time > proc_max_time:
                # If a subprocess has been running for too long, kill it
                # Note no offset file will be written in this case
                pid.kill()
                del SUBPROCESS_LIST[i]
                break

        if len(SUBPROCESS_LIST) <= subprocess_count:
            # A slot opened up! Or all processes finished. Depending on what we're
            # waiting for.
            break
        time.sleep(1)
