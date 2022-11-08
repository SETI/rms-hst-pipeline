#!/usr/bin/env python3
##########################################################################################
# pipeline/query-mast.py
#
# Syntax:
#   pipeline/query-mast.py proposal_id [options] path [path ...]
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from query_mast import get_products_from_mast
from product_labels.suffix_info import ACCEPTED_SUFFIXES, INSTRUMENT_FROM_LETTER_CODE

TWD = os.environ["TMP_WORKING_DIR"]
DEFAULT_DIR = TWD + "/files_from_mast"
LOG_DIR = 'logs'
START_DATE = (1900, 1, 1)
END_DATE = (2025, 1, 1)
RETRY = 1

# Set up parser
parser = argparse.ArgumentParser(
    description='query-mast: Perform mast query with a given proposal id')

parser.add_argument('proposal_id', nargs='+', type=str,
    help='The proposal id for the mast query')

parser.add_argument('--path', '-p', type=str, action='store', default='',
    help='Optional path to store downloaded files from mast.')

parser.add_argument('--start', type=str, action='store', default='',
    help='Optional start date from MAST in (yyyy, mm, dd) format.')

parser.add_argument('--end', type=str, action='store', default='',
    help='Optional end date from MAST in (yyyy, mm, dd) format.')

parser.add_argument('--retry', '-r', type=str, action='store', default='',
    help='Optional max number of Mast connection retry.')

parser.add_argument('--log', '-l', type=str, default='',
    help='Path and name for the log file. The name always has the current date and time '+
         'appended. If not specified, the file will be written to the current working '  +
         'directory and named "label-hst-files-<date>.log".')

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Parse and validate the command line
args = parser.parse_args()

logger = pdslogger.PdsLogger('pds.hst.query-mast')
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'hst-query-mast-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    logpath = LOG_DIR + '/hst-query-mast-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))
LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('query-mast ' + ' '.join(sys.argv[1:]), limits=LIMITS)

start_date = args.start if args.start else START_DATE
end_date = args.end if args.end else END_DATE
retry = args.retry if args.retry else RETRY
dir = args.path if args.path else DEFAULT_DIR

get_products_from_mast(proposal_id = args.proposal_id,
                       start_date = start_date,
                       end_date = end_date,
                       logger = logger,
                       max_retries = retry,
                       dir = dir,
                       testing=False)
logger.close()

##########################################################################################
