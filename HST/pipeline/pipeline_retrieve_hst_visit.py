#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_retrieve_hst_visit.py
#
# Syntax:
# pipeline_retrieve_hst_visit.py [-h] --proposal-id PROPOSAL_ID --visit VISIT
#                                [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform retrieve_hst_visit task to download all identified files.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from hst_helper import HST_DIR
from hst_helper.fs_utils import get_formatted_proposal_id
from retrieve_hst_visit import retrieve_hst_visit
from queue_manager.task_queue_db import (remove_a_task,
                                         remove_all_tasks_for_a_prog_id)

# Set up parser
parser = argparse.ArgumentParser(
    description="""pipeline_retrieve_hst_visit: Retrieve all the identified files for a
                given proposal id and visit.""")

parser.add_argument('--proposal-id', type=str, default='', required=True,
    help='The proposal id for the MAST query.')

parser.add_argument('--visit', '--vi', type=str, default='', required=True,
    help='The two character visit of an observation.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "retrieve-hst-visit-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some params are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
proposal_id = args.proposal_id
visit = args.visit.zfill(2)
LOG_DIR = HST_DIR['pipeline'] + f'/hst_{proposal_id.zfill(5)}/visit_{visit}/logs'

logger = pdslogger.PdsLogger(f'pds.hst.retrieve-hst-visit-{proposal_id}'
                             f'-visit_{visit.zfill(2)}')
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'retrieve-hst-visit-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/retrieve-hst-visit-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('retrieve-hst-visit ' + ' '.join(sys.argv[1:]), limits=LIMITS)
formatted_proposal_id = get_formatted_proposal_id(proposal_id)

try:
    retrieve_hst_visit(proposal_id, visit, logger)
except Exception as e:
    # Before raising the error, remove the task queue of the proposal id from database.
    remove_all_tasks_for_a_prog_id(formatted_proposal_id)
    logger.error(e)
    raise

remove_a_task(formatted_proposal_id, visit, 'retrieve_visit')
logger.close()

##########################################################################################
