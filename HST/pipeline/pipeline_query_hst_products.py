#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_query_hst_products.py
#
# Syntax:
# pipeline_query_hst_products.py [-h] [--log LOG] [--quiet] proposal_id
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from query_hst_products import query_hst_products
from queue_manager import queue_next_task
from hst_helper import HST_DIR

# Set up parser
parser = argparse.ArgumentParser(
    description="""query-hst-products: Perform mast query with a given proposal id and
                download all TRL files for this HST program.""")

parser.add_argument('--proposal_id', '--prog-id', type=str, default='', required=True,
    help='The proposal id for the mast query.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "query-hst-products-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some query constraints are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
proposal_id = args.proposal_id
LOG_DIR = HST_DIR['pipeline'] + '/hst_'  + proposal_id.zfill(5) + '/logs'

logger = pdslogger.PdsLogger('pds.hst.query-hst-products-' + proposal_id)
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'query-hst-products-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/query-hst-products-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('query-hst-products ' + ' '.join(sys.argv[1:]), limits=LIMITS)

new_visit_li, all_visits = query_hst_products(proposal_id, logger)
logger.info('List of visits in which any files are new or chagned: ' + str(new_visit_li))
# TODO: TASK QUEUE
# - if list is not empty, queue update-hst-program with the list of visits
print(f'===========Queue in get_program_info, task: 3 for {proposal_id}===========')
queue_next_task(proposal_id, 'all', 3, logger)
# - if list is empty, re-queue query-hst-products with a 30-day delay
# - re-queue query-hst-products with a 90-day delay


logger.close()

##########################################################################################
