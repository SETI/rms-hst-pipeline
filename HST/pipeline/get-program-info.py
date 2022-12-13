#!/usr/bin/env python3
##########################################################################################
# pipeline/get-program-info.py
#
# Syntax:
# get-program-info.py [-h] [--log LOG] [--quiet] proposal_id
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from get_program_info import get_program_info
from hst_helper import HST_DIR

# Set up parser
parser = argparse.ArgumentParser(
    description='get-program-info: Perform mast query with a given proposal id')

parser.add_argument('proposal_id', type=str, default='',
    help='The proposal id for the mast query')

parser.add_argument('--log', '-l', type=str, default='',
    help='Path and name for the log file. The name always has the current date and time '+
         'appended. If not specified, the file will be written to the current logs '  +
         'directory and named "get-program-info-<date>.log".')

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some query constraints are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

args = parser.parse_args()
proposal_id = args.proposal_id
LOG_DIR = HST_DIR['pipeline'] + '/hst_'  + proposal_id.zfill(5) + '/logs'

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

logger.info('Get program info for proposal id: ' + str(proposal_id))
get_program_info(proposal_id, None, logger)
# TASK QUEUE: None

logger.close()

##########################################################################################