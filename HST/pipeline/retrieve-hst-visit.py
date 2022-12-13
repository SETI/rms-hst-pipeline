#!/usr/bin/env python3
##########################################################################################
# pipeline/retrieve-hst-visit.py
#
# Syntax:
# usage: retrieve-hst-visit.py [-h] [--proposal_id PROPOSAL_ID] [--visit VISIT]
#                              [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from retrieve_hst_visit import retrieve_hst_visit
from hst_helper import HST_DIR

# Set up parser
parser = argparse.ArgumentParser(
    description='retrieve-hst-visit: Perform mast query with a given proposal id')

parser.add_argument('--proposal_id', '-pid', type=str, default='',
    help='The proposal id for the mast query')

parser.add_argument('--visit', '-vi', type=str, default='',
    help='The two character visit of an observation')

parser.add_argument('--log', '-l', type=str, default='',
    help='Path and name for the log file. The name always has the current date and time '+
         'appended. If not specified, the file will be written to the current logs '  +
         'directory and named "retrieve-hst-visit-<date>.log".')

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some query constraints are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
proposal_id = args.proposal_id
visit = args.visit.zfill(2)
LOG_DIR = HST_DIR['pipeline'] + f'/hst_{proposal_id.zfill(5)}/visit_{visit}/logs'

logger = pdslogger.PdsLogger('pds.hst.retrieve-hst-visit-' + proposal_id)
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
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    logpath = LOG_DIR + '/retrieve-hst-visit-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('retrieve-hst-visit ' + ' '.join(sys.argv[1:]), limits=LIMITS)

logger.info(f'Retrieve accepted files for proposal id: {proposal_id} & visit: {visit}')
retrieve_hst_visit(proposal_id, visit, logger)

logger.close()

##########################################################################################