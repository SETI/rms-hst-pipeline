#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_query_hst_moving_targets.py
#
# Syntax:
# pipeline_query_hst_moving_targets.py [-h]
#                                      [--proposal_ids PROPOSAL_IDS [PROPOSAL_IDS ...]]
#                                      [--instruments INSTRUMENTS [INSTRUMENTS ...]]
#                                      [--start START] [--end END]
#                                      [--retry RETRY] [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from query_hst_moving_targets import query_hst_moving_targets
from hst_helper import (START_DATE,
                        END_DATE,
                        RETRY,
                        HST_DIR)

# Set up parser
parser = argparse.ArgumentParser(
    description="""query-hst-moving-targets: Perform mast query with given query
                constraints.""")

parser.add_argument('--proposal_ids', '--prog-id', nargs='+', type=str, default='',
    help='The proposal ids for the mast query')

parser.add_argument('--instruments', '-i', nargs='+', type=str, default='',
    help='The instruments for the mast query.')

parser.add_argument('--start', type=str, action='store', default='',
    help='Optional start date from MAST in (yyyy, mm, dd) format.')

parser.add_argument('--end', type=str, action='store', default='',
    help='Optional end date from MAST in (yyyy, mm, dd) format.')

parser.add_argument('--retry', '-r', type=str, action='store', default='',
    help='Optional max number of Mast connection retry.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current logs
         directory and named "query-hst-moving-targets-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some query constraints are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
LOG_DIR = HST_DIR['pipeline'] + '/logs'

logger = pdslogger.PdsLogger('pds.hst.query-hst-moving-targets')
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'query-hst-moving-targets-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/query-hst-moving-targets-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('query-hst-moving-targets ' + ' '.join(sys.argv[1:]), limits=LIMITS)


proposal_ids = args.proposal_ids if args.proposal_ids else []
instruments = args.instruments if args.instruments else []
start_date = args.start if args.start else START_DATE
end_date = args.end if args.end else END_DATE
retry = args.retry if args.retry else RETRY

logger.info('Mast query constraints: ' + str(args))
pid_li = query_hst_moving_targets(proposal_ids=proposal_ids,
                                  instruments=instruments,
                                  start_date=start_date,
                                  end_date=end_date,
                                  logger=logger,
                                  max_retries=retry)
logger.info('List of program ids: ' + str(pid_li))
# TODO: TASK QUEUE
# - if there is a missing HST_PIPELINE/hst_<nnnnn> missing, queue query-hst-products
# - re-queue query-hst-moving-targets with a 30-day delay

logger.close()

##########################################################################################
