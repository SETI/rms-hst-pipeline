#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_get_program_info.py
#
# Syntax:
# pipeline_get_program_info.py [-h] --proposal-id PROPOSAL_ID [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform get_program_info to retrieve the online files that describe a program (such as
# .apt or .pro) and assemble other program-level information. These are the actions:
#
# - Download the proposal files via a web query.
# - If these files are the same as the existing ones, return.
# - Otherwise:
#   - Rename and back up the existing ones to backups/ subdirectory.
#   - Save the newly downloaded files.
# - Regenerate the new program-info.txt.
##########################################################################################

import argparse
import sys

from hst_pipeline.get_program_info import get_program_info
from hst_pipeline.hst_helper.fs_utils import get_formatted_proposal_id
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)
from hst_pipeline.queue_manager.task_queue_db import remove_a_task


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_get_program_info: Retrieve the proposal files via a web query
                    for a given proposal id.""")

    parser.add_argument('--proposal-id', type=str, default='', required=True,
        help='The proposal id for the MAST query.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "get-program-info-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('get-program-info', args, argv, proposal_id=args.proposal_id)


def main(argv=None):
    """Perform the get_program_info pipeline task."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        proposal_id = args.proposal_id
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)

        try:
            get_program_info(proposal_id, None, logger)
        except Exception as e:
            logger.exception(e)
            raise

        remove_a_task(formatted_proposal_id, '', 'get_prog_info')
    finally:
        close_task_logger(logger)


if __name__ == '__main__':
    main()

##########################################################################################
