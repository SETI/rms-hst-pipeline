#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_dedupe_staging.py
#
# Syntax:
# pipeline_dedupe_staging.py [-h] --proposal-id PROPOSAL_ID [--delete] [--log LOG]
#                            [--quiet]
#
# Enter the --help option to see more information.
#
# Remove the duplicate mastDownload trees that older versions of the pipeline left in the
# staging directory of a proposal. Without --delete, only report what would be removed.
# This is a one-time maintenance command, not a task in the task queue.
##########################################################################################

import argparse
import sys

from hst_pipeline.dedupe_staging import dedupe_staging
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_dedupe_staging: remove each visit's mastDownload tree from
                    the staging directory of a proposal once every file in it is verified
                    to have an identical copy in its collection directory.""")

    parser.add_argument('--proposal-id', type=str, default='', required=True,
        help='The proposal id whose staging directory is deduplicated.')

    parser.add_argument('--delete', action='store_true',
        help="""Delete the verified mastDownload trees. Without this option, only report
             what would be deleted.""")

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "dedupe-staging-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('dedupe-staging', args, argv, proposal_id=args.proposal_id)


def main(argv=None):
    """Run dedupe_staging for one proposal. Exit with status 1 if any visit was kept
    because one of its files could not be verified."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        try:
            results = dedupe_staging(args.proposal_id, delete=args.delete, logger=logger)
        except Exception as e:
            logger.exception(e)
            raise
    finally:
        close_task_logger(logger)

    if any(result.problems for result in results):
        sys.exit(1)


if __name__ == '__main__':
    main()

##########################################################################################
