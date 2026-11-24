#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_run_full_process.py
#
# Syntax:
# pipeline_run_full_process.py [-h] [--proposal-ids PROPOSAL_IDS [PROPOSAL_IDS ...]]
#                 [--log LOG] [--quiet]
#                 [--max-subproc-cnt MAX_SUBPROC_CNT]
#                 [--max-allowed-time MAX_ALLOWED_TIME] [--recreate-queue] [--run-forever]
#
# Enter the --help option to see more information.
#
# The script to start hst pipeline process for the given proposal ids.
##########################################################################################

import argparse

from hst_pipeline import queue_manager
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)
from hst_pipeline.queue_manager import (MAX_SUBPROCESS_CNT,
                                        run_full_process)
from hst_pipeline.queue_manager.task_queue_db import (create_task_queue_table,
                                                      erase_all_task_queue,
                                                      init_task_queue_table)
from sqlalchemy.exc import OperationalError


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_run_full_process: run pipeline with all available ids from MAST or
                    the given proposal ids.""")

    parser.add_argument('--proposal-ids', nargs='*', type=str, default=None,
        help='The specified proposal ids for the pipeline run (omit values to query all).')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "run-pipeline-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    parser.add_argument('--max-subproc-cnt', '--max-subproc',
        type=int, action='store', default=MAX_SUBPROCESS_CNT,
        help='Max number of subprocesses to run at a time for one pipeline process.')

    parser.add_argument('--max-allowed-time', '--max-time',
        type=int, action='store', default=1800,
        help='Max allowed subprocess time in seconds before it gets killed.')

    parser.add_argument('--nocleanup', action='store_true',
        help="Don't delete the MAST download files.")

    parser.add_argument('--recreate-queue', action='store_true',
        help='Clear the task queue before starting the pipeline.')

    parser.add_argument('--run-forever', action='store_true',
        help='Keep running after the task queue is empty (poll for new tasks).')

    return parser.parse_args(argv)


def normalize_proposal_ids(proposal_ids):
    """Return a cleaned proposal id list, or None to query all ids from MAST."""
    if proposal_ids is None:
        return None
    proposal_ids = [p.strip() for p in proposal_ids if p.strip()]
    return proposal_ids or None


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('run-pipeline', args, argv)


def recreate_task_queue(logger):
    """Initialize or reset the task queue table."""
    try:
        init_task_queue_table()
    except OperationalError as e:  # pragma: no cover
        msg = str(e)
        if 'already exists' in msg:
            erase_all_task_queue()
        elif 'no such table' in msg:
            create_task_queue_table()
        else:
            logger.exception('Failed to create task queue table!')
            raise


def main(argv=None):
    """Run the full HST pipeline process for the given proposal ids."""
    args = parse_args(argv)

    queue_manager.MAX_ALLOWED_TIME = args.max_allowed_time
    queue_manager.MAX_SUBPROCESS_CNT = args.max_subproc_cnt

    logger = setup_logger(args, argv=argv)
    try:
        proposal_ids = normalize_proposal_ids(args.proposal_ids)

        if args.recreate_queue:
            recreate_task_queue(logger)

        run_full_process(proposal_ids, logger,
                         run_forever=args.run_forever,
                         no_cleanup=args.nocleanup)
    finally:
        close_task_logger(logger)


if __name__ == '__main__':
    main()

##########################################################################################
