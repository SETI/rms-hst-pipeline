##########################################################################################
# queue_manager/__init__.py
#
# Queue manager module that will queue in the next task for the hst pipeline process.
# - run_pipeline will start a hst pipeline process for the given proposal id list.
# - queue_next_task will Queue in the next task for a given proposal id to database, and
#   wait for the open subprocess slot to execute the corresponding command.
##########################################################################################

import os
import pdslogger
import subprocess
import time

from hst_helper.fs_utils import get_formatted_proposal_id
from queue_manager.task_queue_db import (add_a_task,
                                         create_task_queue_table,
                                         db_exists,
                                         erase_all_task_queue,
                                         get_next_task_to_be_run,
                                         get_total_number_of_tasks,
                                         init_task_queue_table,
                                         update_a_task_status)
from queue_manager.config import (DB_PATH,
                                  HST_SOURCE_ROOT,
                                  PYTHON_EXE,
                                  MAX_ALLOWED_TIME,
                                  MAX_SUBPROCESS_CNT,
                                  SUBPROCESS_LIST,
                                  TASK_INFO)
from sqlalchemy.exc import OperationalError

def run_pipeline(proposal_ids, logger=None):
    """With a given list of proposal ids, run pipeline for each program id.

    Inputs:
        proposal_ids    a list of proposal ids.
        logger          pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Run pipeline with proposal ids: {proposal_ids}')

    try:
        init_task_queue_table()
    except OperationalError as e:
        if 'already exists' in e.__repr__():
            erase_all_task_queue()
        elif 'no such table' in e.__repr__():
            create_task_queue_table()
        else:
            logger.error('Failed to create task queue table!')
            raise Exception('Failed to create task queue table!') # fatal error

    # Kick start the pipeline for each proposal id
    # proc_li = []
    for prog_id in proposal_ids:
        try:
            proposal_id = int(prog_id)
        except ValueError:
            logger.warn(f'Proposal id: {prog_id} is not valid')

        formatted_proposal_id = get_formatted_proposal_id(proposal_id)
        # Start hst pipeline for each proposal id
        logger.info(f'Starting to run pipeline for {proposal_id}')
        logger.info(f'Queue query_hst_moving_targets for {proposal_id}')
        queue_next_task(formatted_proposal_id, '', 'query_moving_targ', logger)

    # spawning subprocesses
    while get_total_number_of_tasks() > 0:
        task = get_next_task_to_be_run()

        if task is not None:
            cmd_parts = task.cmd.split(' ')
            program_path = os.path.join(HST_SOURCE_ROOT, cmd_parts[0])
            sub_args = [PYTHON_EXE, program_path] + cmd_parts[1::]
            max_allowed_time = MAX_ALLOWED_TIME
            run_and_maybe_wait(sub_args, max_allowed_time, task.proposal_id,
                               task.visit, task.task, logger)
            # logger.debug("Spawning subprocess %s", str(sub_args))
        time.sleep(1)

    logger.info('Pipeline complete!')

def queue_next_task(proposal_id, visit_info, task, logger):
    """Queue in the next task for a given proposal id to database.

    1. Update the next task for a given proposal id to database. (task status: 0)
    2. Wait for an open slot in subprocess list, once there is an open slot, update
    the task status to 1 in the database.
    3. Run the task command (spawn the subprocess)

    Inputs:
        proposal_id    the proposal if of the current task.
        visit_info     a two character visit, a list of visits or ''.
        task           a string represents the current task.
        logger         pdslogger to use; None for default EasyLogger.

    Returns:    the child process that executes the given task.
    """
    # if DB doesn't exist, log a warning message and return
    if not db_exists():
        logger.warn(f'Task queue db: {DB_PATH} does not exist')
        return

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Queue in the next task for: {formatted_proposal_id}'
                f', task: {task}, visit: {visit_info}')

    visit_arg = ' '.join(visit_info) if isinstance(visit_info, list) else visit_info
    visit = '' if isinstance(visit_info, list) else visit_info

    priority = TASK_INFO[task][1]
    order = TASK_INFO[task][0]
    cmd = TASK_INFO[task][2].replace('{P}', formatted_proposal_id)
    cmd = cmd.replace('{V}', visit_arg)
    # if the task has been queued, we don't spawn duplicated subprocess.
    spawn_subproc = add_a_task(formatted_proposal_id, visit,
                               task, priority, order, 0, cmd)
    if spawn_subproc is False:
        return

    return True

def run_and_maybe_wait(args, max_allowed_time, proposal_id, visit, task, logger):
    """Run one subprocess, waiting as necessary for a slot to open up.

    Inputs:
        args                the command of a subprocess to be executed.
        max_allowed_time    the max time for a subprocess to be done before killing it.
        proposal_id         the proposal id of the current task.
        visit               two character visit.
        task                the task waiting to be run
        logger              pdslogger to use; None for default EasyLogger.

    Returns:    the child process that executes the given args.
    """
    # wait for an open subprocess slot
    wait_for_subprocess()

    update_a_task_status(proposal_id, visit, task, 1)
    logger.debug("Spawning subprocess", str(args))
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append((pid, time.time(), time.time()+max_allowed_time, proposal_id, args))

def wait_for_subprocess(all=False):
    """Wait for one (or all) subprocess slots to open up.

    Inputs:
        all         a flag to determine if we are waiting for all subprocess slots to
                    open up.
    """
    subprocess_count = MAX_SUBPROCESS_CNT

    if all:
       subprocess_count = 0

    cur_time = time.time()
    while len(SUBPROCESS_LIST) > 0:
        for i in range(len(SUBPROCESS_LIST)):
            pid, _, proc_max_time, _, _ = SUBPROCESS_LIST[i]

            if pid.poll() is not None:
                # The subprocess completed, make the slot available for next subprocess
                del SUBPROCESS_LIST[i]
                break

            if cur_time > proc_max_time and pid:
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
