#!/usr/bin/env python3
##########################################################################################
# pipeline/label-products.py
#
# Syntax:
#   pipeline/label-products.py [options] path [path ...]
#
# Enter the --help option to see more information.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from product_labels import label_hst_fits_directories

# Set up parser
parser = argparse.ArgumentParser(
    description='label-hst: Create and update PDS4 labels for HST data products')

parser.add_argument('path', nargs='+', type=str,
    help='The path to a directory containing a "logically complete" set of HST FITS '    +
         'files downloaded from MAST. "Logically complete" means that every file that '  +
         ' a given file might  need to refer to is also in the same directory. A '       +
         'directory containing all the files from a single HST visit is always '         +
         'logically complete.')

parser.add_argument('--old', type=str, action='store', default='',
    help='Path to another directory tree that might contain older versions of the same ' +
         'products and their labels. If specified, the new labels will increment the '   +
         'version number relative to the old labels, with a new fractional number if '   +
         'the label has changed but the data file is the same, or with a new whole '     +
         'number if the data file has changed.')

parser.add_argument('--select', type=str, action='store', default='',
    help='Optional wildcard pattern. Files that do not match this pattern will be '      +
         'ignored.')

parser.add_argument('--date', type=str, action='store', default='',
    help='Optional retrieval date from MAST in yyyy-mm-dd format. This is used if the '  +
         'product creation date cannot be otherwise inferred from the file.')

parser.add_argument('--replace-nans', '-N', action='store_true',
    help='Replace any floating-point NaNs with a special constant.')

parser.add_argument('--reset-dates', '-D', action='store_true',
    help='Reset file modification dates to match the inferred product creation times.')

parser.add_argument('--log', '-l', type=str, default='',
    help='Path and name for the log file. The name always has the current date and time '+
         'appended. If not specified, the file will be written to the current working '  +
         'directory and named "label-hst-files-<date>.log".')

parser.add_argument('--quiet', '-q', action='store_true',
    help='Do not also log to the terminal.')

# Parse and validate the command line
args = parser.parse_args()

logger = pdslogger.PdsLogger('pds.hst.label-products')
if not args.quiet:
    logger.add_handler(pdslogger.stdout_handler)

# Define the log file
now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
if args.log:
    if os.path.isdir(args.log):
        logpath = os.path.join(args.log, 'hst-label-products-' + now + '.log')
    else:
        parts = os.path.splitext(args.log)
        logpath = parts[0] + '-' + now + parts[1]
else:
    logpath = 'hst-label-products-' + now + '.log'

logger.add_handler(pdslogger.file_handler(logpath))

LIMITS = {'info': -1, 'debug': -1, 'normal': -1}
logger.open('label-products ' + ' '.join(sys.argv[1:]), limits=LIMITS)

label_hst_fits_directories(args.path,
                           match_pattern = args.select,
                           old_directories = [args.old],
                           retrieval_date = args.date,
                           logger = logger,
                           reset_dates = args.reset_dates,
                           replace_nans = args.replace_nans)
logger.close()

##########################################################################################
