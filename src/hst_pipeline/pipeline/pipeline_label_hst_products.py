#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_label_hst_products.py
#
# Syntax:
# pipeline_label_hst_products.py [-h] [--proposal-id PROPOSAL_ID] [--visit VISIT]
#                                [--path PATH] [--old OLD][--select SELECT] [--date DATE]
#                                [--reset-dates] [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform label_hst_products task with these actions:
#
# - Compare the staged FITS files to those in an existing bundle, if any.
# - Create a new XML label for each file.
# - Reset the modification dates of the FITS files to match their production date at MAST.
# - Create a Product_Browse label for each MAST preview.
# - For a visit, replace its products.txt and trl_checksums.txt with the pending lists
#   from query_hst_products, if any (see fs_utils.commit_visit_lists).
#
# With --proposal-id and --visit, it labels the visit's files after the
# prepare_browse_products task has filed them into collection directories under staging.
# With --path, it labels the FITS files under that directory instead.
##########################################################################################

import argparse
import sys

from hst_pipeline.hst_helper.fs_utils import (commit_visit_lists,
                                              get_formatted_proposal_id)
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)
from hst_pipeline.label_hst_products import label_hst_visit
from hst_pipeline.product_labels import label_hst_fits_directories
from hst_pipeline.queue_manager.task_queue_db import remove_a_task


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_label_hst_products: products and update PDS4 labels for HST
                    data products.""")

    parser.add_argument('--proposal-id', type=str, default='',
        help='The proposal id for the MAST query.')

    parser.add_argument('--visit', '--vi', type=str, default='',
        help='The two character visit of an observation.')

    parser.add_argument('--path', type=str, action='store', default='',
        help="""The path to a directory containing a "logically complete" set of HST FITS
             files downloaded from MAST. "Logically complete" means that every file that
             a given file might  need to refer to is also in the same directory. A
             directory containing all the files from a single HST visit is always
             logically complete. If omitted, the files of --visit that have been filed
             into collection directories under the staging directory are labeled.""")

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

    parser.add_argument('--reset-dates', '-D', action='store_true',
        help='Reset file modification dates to match the inferred product creation times.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current working
             directory and named "label-hst-products-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def check_target(args):
    """Raise a ValueError unless --path or both --proposal-id and --visit are given."""
    if not args.path and not (args.proposal_id and args.visit):
        raise ValueError(
            'Either --path or both --proposal-id and --visit must be provided.'
        )


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('label-hst-products', args, argv,
                             proposal_id=args.proposal_id, visit=args.visit, visit_logs=True)


def main(argv=None):
    """Perform the label_hst_products pipeline task."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        proposal_id = args.proposal_id
        visit = args.visit.zfill(2) if args.visit else ''
        check_target(args)
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)

        try:
            if args.path:
                label_hst_fits_directories(args.path,
                                           match_pattern=args.select,
                                           old_directories=[args.old],
                                           retrieval_date=args.date,
                                           logger=logger,
                                           reset_dates=args.reset_dates)
            else:
                label_hst_visit(proposal_id, visit,
                                match_pattern=args.select,
                                old_directories=[args.old],
                                retrieval_date=args.date,
                                logger=logger,
                                reset_dates=args.reset_dates)
                # The visit is now up to date with the query that found it changed
                for path in commit_visit_lists(proposal_id, visit):
                    logger.info('Visit list updated', path)
        except Exception as e:
            logger.exception(e)
            raise

        remove_a_task(formatted_proposal_id, visit, 'label_prod')
    finally:
        close_task_logger(logger)


if __name__ == '__main__':
    main()

##########################################################################################
