#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_get_program_info.py
#
# Syntax:
# pipeline_get_program_info.py [-h] [--log LOG] [--quiet] proposal_id
#
# Enter the --help option to see more information.
#
# Perform get_program_info to retrieve the online files that describe a program (such as
# .apt or .pro) and assemble other program-level information. These are the actions:
#
# - Download the proposal files via a web query.
# - If these files are the same as the existing ones, return.
# - Otherwise:
#   - Rename and back up the existing ones to backups/ subdirectory.
#   - Save the newly downloaded files.
# - Regenerate the new program-info.txt.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from get_program_info import get_program_info
from hst_helper import HST_DIR
from hst_helper.fs_utils import get_formatted_proposal_id
from queue_manager.task_queue_db import remove_all_task_queue_for_a_prog_id

# Set up parser
parser = argparse.ArgumentParser(
    description="""pipeline_get_program_info: Retrieve the proposal files via a web query
                for a given proposal id.""")

parser.add_argument('--proposal_id', '--prog-id', type=str, default='', required=True,
    help='The proposal id for the MAST query.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "get-program-info-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some params are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

args = parser.parse_args()
proposal_id = args.proposal_id
LOG_DIR = f'{HST_DIR["pipeline"]}/hst_{proposal_id.zfill(5)}/logs'

logger = pdslogger.PdsLogger('pds.hst.get-program-info-' + proposal_id)
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'get-program-info-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/get-program-info-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('get-program-info ' + ' '.join(sys.argv[1:]), limits=LIMITS)

try:
    get_program_info(proposal_id, None, logger)
except:
    # Before raising the error, remove the task queue of the proposal id from database.
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    remove_all_task_queue_for_a_prog_id(formatted_proposal_id)
    raise

logger.close()

##########################################################################################
