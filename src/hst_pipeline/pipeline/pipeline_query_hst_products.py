#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_query_hst_products.py
#
# Syntax:
# pipeline_query_hst_products.py [-h] --proposal-id PROPOSAL_ID [--log LOG]
#                                [--quiet] [--taskqueue]
#
# Enter the --help option to see more information.
#
# Perform query_hst_products task with these actions:
#
# - Query MAST to get:
#   - a complete list of the accepted files for a given proposal id. Update or create
#     products.txt in <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/.
#   - a list of all TRL files and their checksums. Update or create trl_checksums.txt
#     in <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/.
#   - Return a list of visits with changed or new files.
#   - Queue update_hst_program if the list of visits with changed or new files is not
#     emtpy.
##########################################################################################

import argparse
import sys

from hst_pipeline.hst_helper.fs_utils import (get_formatted_proposal_id,
                                              get_program_dir_path)
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)
from hst_pipeline.query_hst_products import query_hst_products
from hst_pipeline.queue_manager import queue_next_task
from hst_pipeline.queue_manager.task_queue_db import remove_a_task


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_query_hst_products: Perform MAST query with a given proposal
                    id and download all TRL files for this HST program.""")

    parser.add_argument('--proposal-id', type=str, default='', required=True,
        help='The proposal id for the MAST query.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "query-hst-products-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    parser.add_argument('--taskqueue', '--tq', action='store_true',
        help='Run the script with task queue.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('query-hst-products', args, argv,
                             proposal_id=args.proposal_id)


def queue_follow_up_tasks(proposal_id, formatted_proposal_id, new_visit_li, logger):
    """Queue update_hst_program or log completion, then remove query_prod task."""
    if len(new_visit_li) != 0:
        logger.info(f'Queue update_hst_program for {proposal_id}')
        queue_next_task(formatted_proposal_id, new_visit_li, 'update_prog', logger)
    else:
        staging_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
        logger.info(f'No new or changed files, {staging_dir} is fully populated.'
                    + f' Pipeline stops for {proposal_id}')

    remove_a_task(formatted_proposal_id, '', 'query_prod')


def main(argv=None):
    """Perform the query_hst_products pipeline task."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        proposal_id = args.proposal_id
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)

        try:
            new_visit_li, _ = query_hst_products(proposal_id, logger)
            logger.info(f'List of visits for {proposal_id} in which any files are new or changed: '
                        + str(new_visit_li))
        except Exception as e:
            logger.exception(e)
            raise

        if args.taskqueue:
            queue_follow_up_tasks(proposal_id, formatted_proposal_id, new_visit_li, logger)
    finally:
        close_task_logger(logger)


if __name__ == '__main__':
    main()

##########################################################################################
