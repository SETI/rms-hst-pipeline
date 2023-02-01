#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_finalize_hst_bundle.py
#
# Syntax:
# pipeline_finalize_hst_bundle.py [-h] --proposal_id PROPOSAL_ID --visit
#                                     VISIT [--log LOG] [--quiet]
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from finalize_hst_bundle import finalize_hst_bundle
from hst_helper import HST_DIR

# Set up parser
parser = argparse.ArgumentParser(
    description="""finalize-hst-bundle: package a complete set of files in the staging
    directories as a new bundle or as updates to an existing bundle.""")

parser.add_argument('--proposal_id', '-pid', type=str, default='', required=True,
    help='The proposal id for the mast query.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "finalize-hst-bundle-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some params are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
proposal_id = args.proposal_id
LOG_DIR = HST_DIR['pipeline'] + f'/hst_{proposal_id.zfill(5)}/logs'

logger = pdslogger.PdsLogger('pds.hst.finalize-hst-bundle-' + proposal_id)
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'finalize-hst-bundle-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/finalize-hst-bundle-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('finalize-hst-bundle ' + ' '.join(sys.argv[1:]), limits=LIMITS)

logger.info(f'Finalize hst bundle for proposal id: {proposal_id}')
finalize_hst_bundle(proposal_id, logger)

logger.close()

##########################################################################################
