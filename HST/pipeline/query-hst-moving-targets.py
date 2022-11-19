#!/usr/bin/env python3
##########################################################################################
# pipeline/query-hst-moving-targets.py
#
# Syntax:
#   pipeline/query-hst-moving-targets.py proposal_id [options] path [path ...]
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from query_hst_moving_targets import query_hst_moving_targets
from product_labels.suffix_info import ACCEPTED_SUFFIXES, INSTRUMENT_FROM_LETTER_CODE

# TWD = os.environ["HST_STAGING"]
# DEFAULT_DIR = TWD + "/files_from_mast"
LOG_DIR = os.environ["HST_STAGING"] + '/logs'
START_DATE = (1900, 1, 1)
END_DATE = (2025, 1, 1)
RETRY = 1

# Set up parser
parser = argparse.ArgumentParser(
    description='query-hst-moving-targets: Perform mast query with a given proposal id')

parser.add_argument('proposal_ids', nargs='+', type=str,
    help='The proposal ids for the mast query')

parser.add_argument('--instruments', nargs='+', type=str, default='',
    help='The instruments for the mast query')

parser.add_argument('--start', type=str, action='store', default='',
    help='Optional start date from MAST in (yyyy, mm, dd) format.')

parser.add_argument('--end', type=str, action='store', default='',
    help='Optional end date from MAST in (yyyy, mm, dd) format.')

parser.add_argument('--retry', '-r', type=str, action='store', default='',
    help='Optional max number of Mast connection retry.')

parser.add_argument('--log', '-l', type=str, default='',
    help='Path and name for the log file. The name always has the current date and time '+
         'appended. If not specified, the file will be written to the current logs '  +
         'directory and named "query-hst-moving-targets-<date>.log".')

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Parse and validate the command line
args = parser.parse_args()

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
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    logpath = LOG_DIR + '/query-hst-moving-targets-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('query-hst-moving-targets ' + ' '.join(sys.argv[1:]), limits=LIMITS)


proposal_ids = args.proposal_ids if args.proposal_ids else []
instruments = args.instruments if args.instruments else []
start_date = args.start if args.start else START_DATE
end_date = args.end if args.end else END_DATE
retry = args.retry if args.retry else RETRY
print("##########")
print(args)
li = query_hst_moving_targets(proposal_ids = proposal_ids,
                       instruments = instruments,
                       start_date = start_date,
                       end_date = end_date,
                       logger = logger,
                       max_retries = retry)
print("=============")
print(li)
logger.close()

##########################################################################################