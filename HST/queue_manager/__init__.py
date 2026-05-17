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
import sys
import time

from hst_helper.fs_utils import get_formatted_proposal_id
from queue_manager.task_queue_db import (add_a_task,
                                         create_task_queue_table,
                                         db_exists,
                                         get_next_task_to_be_run,
                                         get_total_number_of_tasks,
                                         queue_cleanup_during_restart,
                                         remove_a_task,
                                         update_a_task_status)
from queue_manager.config import (DB_PATH,
                                  HST_SOURCE_ROOT,
                                  HEARTBEAT_INTERVAL,
                                  PYTHON_EXE,
                                  MAX_ALLOWED_TIME,
                                  MAX_SUBPROCESS_CNT,
                                  REQUEUE_TIME,
                                  SUBPROCESS_LIST,
                                  TASK_INFO)

def run_pipeline(proposal_ids=None, logger=None, run_forever=False):
    """With a given list of proposal ids, run pipeline for each program id.

    Inputs:
        proposal_ids    a list of proposal ids, or None.
        logger          pdslogger to use; None for default EasyLogger.
        run_forever     if True, keep polling the task queue after it becomes empty; if
                        False, exit when no tasks remain.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Run pipeline')

    if not db_exists():
        create_task_queue_table()
    else:
        queue_cleanup_during_restart()

    # if there is no preserved task in the queue manager,
    if get_total_number_of_tasks() == 0:
        if proposal_ids is None:
            logger.info('Start pipeline with all available proposal ids')
            logger.info('Queue query_hst_moving_targets for all available proposal ids')
            queue_next_task('', '', 'query_moving_targ', logger)
        else:
            logger.info(f'Start pipeline with proposal ids: {proposal_ids}')
            for prog_id in proposal_ids:
                try:
                    proposal_id = int(prog_id)
                except ValueError: #pragma: no cover
                    logger.warn(f'Proposal id: {prog_id} is not valid')
                    continue

                formatted_proposal_id = get_formatted_proposal_id(proposal_id)
                # Start hst pipeline for each proposal id
                logger.info(f'Queue query_hst_moving_targets for {proposal_id}')
                queue_next_task(formatted_proposal_id, '', 'query_moving_targ', logger)
    else:
        logger.info(f'Resume pipeline with existing tasks')

    pipeline_start_time = time.time()
    next_requeue_at = pipeline_start_time + REQUEUE_TIME
    last_heartbeat_at = pipeline_start_time

    # spawning subprocesses
    while True:
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
        if get_total_number_of_tasks() > 0 or run_forever:
            if get_total_number_of_tasks() == 0 and run_forever:
                now = time.time()
                if now >= next_requeue_at:
                    if proposal_ids is None:
                        logger.info('Re-queue query_hst_moving_targets for all available '
                                    'proposal ids')
                        queue_next_task('', '', 'query_moving_targ', logger)
                    else:
                        for prog_id in proposal_ids:
                            try:
                                proposal_id = int(prog_id)
                            except ValueError: #pragma: no cover
                                logger.warn(f'Proposal id: {prog_id} is not valid')
                                continue
                            formatted_proposal_id = get_formatted_proposal_id(proposal_id)
                            logger.info(f'Re-queue query_hst_moving_targets for '
                                        f'{proposal_id}')
                            queue_next_task(formatted_proposal_id, '',
                                            'query_moving_targ', logger)
                    next_requeue_at = now + REQUEUE_TIME
                elif now - last_heartbeat_at >= HEARTBEAT_INTERVAL:
                    secs_until_requeue = max(0, int(next_requeue_at - now))
                    print(f'Waiting for next queue (~{secs_until_requeue}s until '
                          f're-queue)...', flush=True)
                    last_heartbeat_at = now
            continue
        else:
            break

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
    if not db_exists(): #pragma: no cover
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
    wait_for_subprocess(logger)

    update_a_task_status(proposal_id, visit, task, 1)
    logger.debug("Spawning subprocess", str(args))
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append((pid, time.time(), time.time()+max_allowed_time,
                            proposal_id, visit, task, args))

def wait_for_subprocess(logger, all=False):
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
            pid, _, proc_max_time, proposal_id, vi, task, args = SUBPROCESS_LIST[i]
            if pid.poll() is not None:
                # The subprocess completed, make the slot available for next subprocess
                del SUBPROCESS_LIST[i]
                logger.info(f'Remove completed subprocess for: {proposal_id}'
                            f', args: {args}')
                break

            if cur_time > proc_max_time and pid:
                # If a subprocess has been running for too long, kill it
                # Note no offset file will be written in this case
                pid.kill()
                del SUBPROCESS_LIST[i]
                logger.info('Remove subprocess running too long '
                            f'(over {MAX_ALLOWED_TIME} seconds, possible hang) '
                            f'for: {proposal_id}, args: {args}')
                # Remove hung task in the queue
                formatted_proposal_id = get_formatted_proposal_id(proposal_id)
                remove_a_task(formatted_proposal_id, vi, task)
                break

        non_wrapper_subproc_cnt = 0
        for sub in SUBPROCESS_LIST:
            task = sub[5]
            if TASK_INFO[task][3] is False:
                non_wrapper_subproc_cnt += 1

        # Count only non-wrapper tasks. This avoids deadlock-like idling when all slots are
        # occupied by wrapper tasks while their spawned subtasks (non-wrapper tasks) cannot
        # start running.
        if non_wrapper_subproc_cnt <= subprocess_count:
            # A slot opened up! Or all processes finished. Depending on what we're
            # waiting for.
            break
        time.sleep(1)
