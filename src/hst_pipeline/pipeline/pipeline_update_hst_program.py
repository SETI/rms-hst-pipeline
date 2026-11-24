#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_update_hst_program.py
#
# Syntax:
# pipeline_update_hst_program.py [-h] --proposal-id PROPOSAL_ID --visits
#                                VISITS [VISITS ...] [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform update_hst_program task with these actions:
#
# - Queue get_program_info and wait for it to complete.
# - Queue update_hst_visit and wait until all visits have completed.
# - Queue finialize_hst_bundle and wait for it to complete.
##########################################################################################

import argparse
import sys

from hst_pipeline.hst_helper.fs_utils import get_formatted_proposal_id
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)
from hst_pipeline.queue_manager.task_queue_db import remove_a_task
from hst_pipeline.update_hst_program import update_hst_program


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_update_hst_program: Update all the identified files for a
                    given proposal id and visit.""")

    parser.add_argument('--proposal-id', type=str, default='', required=True,
        help='The proposal id for the MAST query.')

    parser.add_argument('--visits', nargs='+', type=str, default='', required=True,
        help='A list of the two character visits of an observation.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "update-hst-program-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('update-hst-program', args, argv,
                             proposal_id=args.proposal_id)


def main(argv=None):
    """Perform the update_hst_program pipeline task."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        proposal_id = args.proposal_id
        visits = args.visits if args.visits else []
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)

        try:
            update_hst_program(formatted_proposal_id, visits, logger)
        except Exception:
            logger.exception(
                f'update_hst_program failed for proposal id: {formatted_proposal_id}'
            )
            raise

        visit = '' if isinstance(visits, list) else visits
        remove_a_task(formatted_proposal_id, visit, 'update_prog')
    finally:
        close_task_logger(logger)


if __name__ == '__main__':
    main()

##########################################################################################
