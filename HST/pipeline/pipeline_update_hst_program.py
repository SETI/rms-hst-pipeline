#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_update_hst_program.py
#
# Syntax:
# pipeline_update_hst_program.py [-h] --proposal_id PROPOSAL_ID --visit VISIT
#                              [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform update_hst_program task with these actions:
#
# - Queue get_program_info and wait for it to complete.
# - Queue update_hst_visit and wait until all visits have completed.
# - Queue finialize_hst_bundle and wait for it to complete.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from hst_helper import HST_DIR
from hst_helper.fs_utils import get_formatted_proposal_id
from queue_manager.task_queue_db import (remove_a_task,
                                         remove_all_tasks_for_a_prog_id)
from update_hst_program import update_hst_program

# Set up parser
parser = argparse.ArgumentParser(
    description="""pipeline_update_hst_program: Update all the identified files for a
                given proposal id and visit.""")

parser.add_argument('--proposal-id', type=str, default='', required=True,
    help='The proposal id for the MAST query.')

parser.add_argument('--visit-list', nargs='+', type=str, default='', required=True,
    help='A list of the two character visits of an observation.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "update-hst-program-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some params are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
proposal_id = args.proposal_id
visit_li = args.visit_li if args.visit_li else []
LOG_DIR = f'{HST_DIR["pipeline"]}/hst_{proposal_id.zfill(5)}/logs'

logger = pdslogger.PdsLogger('pds.hst.update-hst-program-' + proposal_id)
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'update-hst-program-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/update-hst-program-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('update-hst-program ' + ' '.join(sys.argv[1:]), limits=LIMITS)
formatted_proposal_id = get_formatted_proposal_id(proposal_id)

try:
    # TODO: uncomment this after debugging
    update_hst_program(formatted_proposal_id, visit_li, logger)
except:
    # Before raising the error, remove the task queue of the proposal id from database.
    remove_all_tasks_for_a_prog_id(formatted_proposal_id)
    raise

visit = '' if isinstance(visit_li, list) else visit_li
remove_a_task(formatted_proposal_id, visit, 'update_prog')
logger.close()

##########################################################################################
