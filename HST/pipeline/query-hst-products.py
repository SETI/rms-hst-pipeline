#!/usr/bin/env python3
##########################################################################################
# pipeline/query-hst-products.py
#
# Syntax:
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from query_hst_products import query_hst_products
from hst_helper import (START_DATE,
                         END_DATE,
                         RETRY,
                         HST_DIR)

# Set up parser
parser = argparse.ArgumentParser(
    description='query-hst-moving-targets: Perform mast query with a given proposal id')

parser.add_argument('proposal_ids', type=str, default='',
    help='The proposal ids for the mast query')

# parser.add_argument('proposal_ids', nargs='+', type=str,
#     help='The proposal ids for the mast query')

# parser.add_argument('--instruments', '-inst', nargs='+', type=str, default='',
#     help='The instruments for the mast query')

# parser.add_argument('--start', type=str, action='store', default='',
#     help='Optional start date from MAST in (yyyy, mm, dd) format.')

# parser.add_argument('--end', type=str, action='store', default='',
#     help='Optional end date from MAST in (yyyy, mm, dd) format.')

# parser.add_argument('--retry', '-r', type=str, action='store', default='',
#     help='Optional max number of Mast connection retry.')

parser.add_argument('--log', '-l', type=str, default='',
    help='Path and name for the log file. The name always has the current date and time '+
         'appended. If not specified, the file will be written to the current logs '  +
         'directory and named "query-hst-moving-targets-<date>.log".')

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some query constraints are passed in
# if len(sys.argv) == 1:
#     parser.print_help()
#     parser.exit()
# Parse and validate the command line
args = parser.parse_args()
proposal_id = args.proposal_ids
LOG_DIR = HST_DIR["pipeline"] + '/hst_'  + proposal_id.zfill(5) + '/logs'

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
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    logpath = LOG_DIR + '/query-hst-products-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('query-hst-products ' + ' '.join(sys.argv[1:]), limits=LIMITS)

logger.info("Query hst products for proposal id: " + str(proposal_id))
visit_li = query_hst_products(proposal_id = proposal_id,
                              logger = logger)
logger.info("List of visits in which any files are new or chagned: " + str(visit_li))
# TODO: TASK QUEUE
# - if list is not empty, queue update-hst-program with the list of visits
# - if list is empty, re-queue query-hst-products with a 30-day delay
# - re-queue query-hst-products with a 90-day delay


logger.close()

##########################################################################################