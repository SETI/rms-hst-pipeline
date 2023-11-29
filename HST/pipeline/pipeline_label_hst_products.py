#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_label_hst_products.py
#
# Syntax:
# pipeline_label_hst_products.py [-h] [--proposal_id PROPOSAL_ID] [--visit VISIT]
#                                [--path PATH] [--old OLD] [--select SELECT]
#                                [--date DATE] [--replace-nans] [--reset-dates]
#                                [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform label_hst_products task with these actions:
#
# - Compare the staged FITS files to those in an existing bundle, if any.
# - Create a new XML label for each file.
# - Reset the modification dates of the FITS files to match their production date at MAST.
# - If any file contains NaNs, rename the original file with “-original” appended,
#   and then rewrite the file without NaNs.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from hst_helper import HST_DIR
from hst_helper.fs_utils import get_formatted_proposal_id
from product_labels import label_hst_fits_directories
from queue_manager.task_queue_db import (remove_a_task,
                                         remove_all_tasks_for_a_prog_id)

# Set up parser
parser = argparse.ArgumentParser(
    description="""pipeline_label_hst_products:productste and update PDS4 labels for HST
                data products.""")

parser.add_argument('--proposal-id', '--prog-id', type=str, default='',
    help='The proposal id for the MAST query.')

parser.add_argument('--visit', '--vi', type=str, default='',
    help='The two character visit of an observation.')

parser.add_argument('--path', type=str, action='store', default='',
    help="""The path to a directory containing a "logically complete" set of HST FITS
         files downloaded from MAST. "Logically complete" means that every file that
         a given file might  need to refer to is also in the same directory. A
         directory containing all the files from a single HST visit is always
         logically complete.""")

parser.add_argument('--old', type=str, action='store', default='',
    help="""Path to another directory tree that might contain older versions of the same
         products and their labels. If specified, the new labels will increment the
         version number relative to the old labels, with a new fractional number if
         the label has changed but the data file is the same, or with a new whole
         number if the data file has changed.""")

parser.add_argument('--select', type=str, action='store', default='',
    help="""Optional wildcard pattern. Files that do not match this pattern will be
         ignored.""")

parser.add_argument('--date', type=str, action='store', default='',
    help="""Optional retrieval date from MAST in yyyy-mm-dd format. This is used if the
         product creation date cannot be otherwise inferred from the file.""")

parser.add_argument('--replace-nans', '-N', action='store_true',
    help='Replace any floating-point NaNs with a special constant.')

parser.add_argument('--reset-dates', '-D', action='store_true',
    help='Reset file modification dates to match the inferred product creation times.')

parser.add_argument('--log', '-l', type=str, default='',
    help="""Path and name for the log file. The name always has the current date and time
         appended. If not specified, the file will be written to the current working
         directory and named "label-hst-products-<date>.log".""")

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Make sure some params are passed in
if len(sys.argv) == 1:
    parser.print_help()
    parser.exit()

# Parse and validate the command line
args = parser.parse_args()
target_path = args.path
proposal_id = args.proposal_id
visit = args.visit

LOG_DIR = (HST_DIR['pipeline'] + '/hst_' + proposal_id.zfill(5) + '/visit_'
           + visit.zfill(2) + '/logs')

# If proposal id and visit are both passed in, it will look for fits files under the
# staging directory for that specific proposal id and visit. Otherwise it will look for
# passed in path.
# TBD: Do we want to set both proposal id and visit as the required params?
if proposal_id and visit and not target_path:
    target_path = (HST_DIR['staging'] + '/hst_' + proposal_id.zfill(5) + '/visit_'
                   + visit.zfill(2))

logger = pdslogger.PdsLogger('pds.hst.label-hst-products')
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'label-hst-products-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    os.makedirs(LOG_DIR, exist_ok=True)
    logpath = LOG_DIR + '/label-hst-products-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))

LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('label-hst-products ' + ' '.join(sys.argv[1:]), limits=LIMITS)
formatted_proposal_id = get_formatted_proposal_id(proposal_id)

try:
    label_hst_fits_directories(target_path,
                               match_pattern = args.select,
                               old_directories = [args.old],
                               retrieval_date = args.date,
                               logger = logger,
                               reset_dates = args.reset_dates,
                               replace_nans = args.replace_nans)
except:
    # Before raising the error, remove the task queue of the proposal id from database.
    remove_all_tasks_for_a_prog_id(formatted_proposal_id)
    raise

remove_a_task(formatted_proposal_id, visit, 'label_prod')
logger.close()

##########################################################################################
