#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_prepare_browse_products.py
#
# Syntax:
# pipeline_prepare_browse_products.py [-h] --proposal-id PROPOSAL_ID --visit VISIT
#                                     [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform prepare_browse_products task: move the files downloaded for a visit from
# <HST_STAGING>/hst_<nnnnn>/visit_<ss>/mastDownload/ into their collection directories,
# <HST_STAGING>/hst_<nnnnn>/<collection>/visit_<ss>/, so each file is stored once. The
# label_hst_products task labels them afterward.
##########################################################################################

import argparse
import sys

from hst_pipeline.hst_helper.fs_utils import get_formatted_proposal_id
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)
from hst_pipeline.prepare_browse_products import prepare_browse_products
from hst_pipeline.queue_manager.task_queue_db import remove_a_task


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_prepare_browse_products: Move the files downloaded for a
                    visit into their collection directories under the staging folder.""")

    parser.add_argument('--proposal-id', type=str, default='', required=True,
        help='The proposal id for the MAST query.')

    parser.add_argument('--visit', '--vi', type=str, default='', required=True,
        help='The two character visit of an observation.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "prepare-browse-products-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('prepare-browse-products', args, argv,
                             proposal_id=args.proposal_id, visit=args.visit, visit_logs=True)


def main(argv=None):
    """Perform the prepare_browse_products pipeline task."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        proposal_id = args.proposal_id
        visit = args.visit.zfill(2)
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)

        try:
            prepare_browse_products(proposal_id, visit, logger)
        except Exception as e:
            logger.exception(e)
            raise

        remove_a_task(formatted_proposal_id, visit, 'prep_browse_prod')
    finally:
        close_task_logger(logger)


if __name__ == '__main__':
    main()

##########################################################################################
