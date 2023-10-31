#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_prepare_browse_products.py
#
# Syntax:
# pipeline_prepare_browse_products.py [-h] --proposal_id PROPOSAL_ID --visit
#                                     VISIT [--log LOG] [--quiet]
# Enter the --help option to see more information.
#
# Perform prepare_browse_products task to prepare the browse products and their labels,
# and save them in Files and labels are in <HST_STAGING>/hst_<nnnnn>/visit_<ss>/.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from hst_helper import HST_DIR
from hst_helper.fs_utils import get_formatted_proposal_id
from prepare_browse_products import prepare_browse_products
from queue_manager.task_queue_db import (remove_a_subprocess_by_prog_id_task_and_visit,
                                         remove_all_subprocess_for_a_prog_id,
                                         remove_all_task_queue_for_a_prog_id)

# Set up parser
parser = argparse.ArgumentParser(
    description="""pipeline_prepare_browse_products: Prepare the browse products and their
                labels and save them in the corresponding staging folders.""")

parser.add_argument('--proposal_id', '--prog-id', type=str, default='', required=True,
    help='The proposal id for the MAST query.')

parser.add_argument('--visit', '--vi', type=str, default='', required=True,
    help='The two character visit of an observation.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "prepare-browse-products-<date>.log".""")

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
LOG_DIR = f'{HST_DIR["pipeline"]}/hst_{proposal_id.zfill(5)}/visit_{visit}/logs'

logger = pdslogger.PdsLogger(f'pds.hst.prepare-browse-products-{proposal_id}'
                             f'-visit_{visit.zfill(2)}')
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'prepare-browse-products-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/prepare-browse-products-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('prepare-browse-products ' + ' '.join(sys.argv[1:]), limits=LIMITS)

try:
    prepare_browse_products(proposal_id, visit, logger)
    remove_a_subprocess_by_prog_id_task_and_visit(proposal_id, 7, visit)
except:
    # Before raising the error, remove the task queue of the proposal id from database.
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    remove_all_task_queue_for_a_prog_id(formatted_proposal_id)
    remove_all_subprocess_for_a_prog_id(formatted_proposal_id)
    raise

logger.close()

##########################################################################################
