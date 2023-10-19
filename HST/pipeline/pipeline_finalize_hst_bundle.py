#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_finalize_hst_bundle.py
#
# Syntax:
# pipeline_finalize_hst_bundle.py [-h] --proposal_id PROPOSAL_ID [--log LOG]
#                                 [--quiet]
# Enter the --help option to see more information.
#
# Perform finalize_hst_bundle task to package a complete set of files in the staging
# directories as a new bundle or as updates to an existing bundle. All bundle files
# will be stored in the bundles directories.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from finalize_hst_bundle import finalize_hst_bundle
from hst_helper import HST_DIR
from hst_helper.fs_utils import get_formatted_proposal_id
from queue_manager.task_queue_db import (remove_a_subprocess_by_prog_id_task_and_visit,
                                         remove_all_subprocess_for_a_prog_id,
                                         remove_all_task_queue_for_a_prog_id)

# Set up parser
parser = argparse.ArgumentParser(
    description="""finalize-hst-bundle: package a complete set of files in the staging
    directories as a new bundle or as updates to an existing bundle.""")

parser.add_argument('--proposal_id', '--prog-id', type=str, default='', required=True,
    help='The proposal id for the MAST query.')

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
LOG_DIR = f'{HST_DIR["pipeline"]}/hst_{proposal_id.zfill(5)}/logs'

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

try:
    finalize_hst_bundle(proposal_id, logger)
    remove_a_subprocess_by_prog_id_task_and_visit(proposal_id, 8, '')
except:
    # Before raising the error, remove the task queue of the proposal id from database.
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    remove_all_task_queue_for_a_prog_id(formatted_proposal_id)
    remove_all_subprocess_for_a_prog_id(formatted_proposal_id)
    raise

logger.close()

##########################################################################################
