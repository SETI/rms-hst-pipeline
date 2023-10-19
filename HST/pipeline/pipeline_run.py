#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_run.py
#
# Syntax:
# pipeline_run.py [-h] --proposal_id PROPOSAL_ID [--log LOG]
#                                 [--quiet]
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

# Set up parser
parser = argparse.ArgumentParser(
    description="""run-pipeline: package a complete set of files in the staging
    directories as a new bundle or as updates to an existing bundle.""")

parser.add_argument('--proposal_ids', '--prog-id', nargs='+', type=str, default='', required=True,
    help='The proposal id for the MAST query.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "run-pipeline-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some params are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
proposal_ids = args.proposal_ids if args.proposal_ids else []
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

run_pipeline(proposal_ids, logger)

logger.close()

##########################################################################################
