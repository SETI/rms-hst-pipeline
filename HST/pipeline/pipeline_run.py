#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_run.py
#
# Syntax:
# pipeline_run.py [-h] [--proposal-ids PROPOSAL_IDS [PROPOSAL_IDS ...]]
#                 [--log LOG] [--quiet]
#                 [--max-subproc-cnt MAX_SUBPROC_CNT]
#                 [--max-allowed-time MAX_ALLOWED_TIME] [--recreate-queue] [--run-forever]
#
# Enter the --help option to see more information.
#
# The script to start hst pipeline process for the given proposal ids.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from hst_helper import HST_DIR
from queue_manager import run_pipeline
from queue_manager.task_queue_db import (create_task_queue_table,
                                         erase_all_task_queue,
                                         init_task_queue_table)
import queue_manager

from sqlalchemy.exc import OperationalError

# Set up parser
parser = argparse.ArgumentParser(
    description="""pipeline_run: run pipeline with all available ids from MAST or the given
                proposal ids.""")

parser.add_argument('--proposal-ids', nargs='*', type=str, default=None,
    help='The specified proposal ids for the pipeline run (omit values to query all).')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "run-pipeline-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

parser.add_argument('--max-subproc-cnt', '--max-subproc',
    type=int, action='store', default=20,
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

# Parse and validate the command line
args = parser.parse_args()

queue_manager.MAX_ALLOWED_TIME = args.max_allowed_time
queue_manager.MAX_SUBPROCESS_CNT = args.max_subproc_cnt

LOG_DIR = HST_DIR['pipeline'] + '/logs'

logger = pdslogger.PdsLogger('pds.hst.run-pipeline')
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'run-pipeline-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/run-pipeline-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('run-pipeline ' + ' '.join(sys.argv[1:]), limits=LIMITS)

# None means query all proposal ids: flag omitted, --proposal-ids with no values,
# or only empty/whitespace ids.
if args.proposal_ids is None:
    proposal_ids = None
else:
    proposal_ids = [p.strip() for p in args.proposal_ids if p.strip()]
    if not proposal_ids:
        proposal_ids = None

if args.recreate_queue:
    try:
        init_task_queue_table()
    except OperationalError as e: #pragma: no cover
        if 'already exists' in repr(e):
            erase_all_task_queue()
        elif 'no such table' in repr(e):
            create_task_queue_table()
        else:
            logger.error('Failed to create task queue table!')
            raise Exception('Failed to create task queue table!') # fatal error

run_pipeline(proposal_ids, logger, run_forever=args.run_forever, no_cleanup=args.nocleanup)

logger.close()

##########################################################################################
