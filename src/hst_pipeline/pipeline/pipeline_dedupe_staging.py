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
import datetime
import os
import pdslogger
import sys

from hst_pipeline.dedupe_staging import dedupe_staging
from hst_pipeline.hst_helper import HST_DIR


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
    """Configure and open the pipeline logger."""
    proposal_id = args.proposal_id
    log_dir = f'{HST_DIR["pipeline"]}/hst_{proposal_id.zfill(5)}/logs'

    logger = pdslogger.PdsLogger('pds.hst.dedupe-staging-' + proposal_id)
    if not args.quiet:
        logger.add_handler(pdslogger.stdout_handler)

    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if args.log:
        if os.path.isdir(args.log):
            logpath = os.path.join(args.log, 'dedupe-staging-' + now + '.log')
        else:
            parts = os.path.splitext(args.log)
            logpath = parts[0] + '-' + now + parts[1]
    else:
        os.makedirs(log_dir, exist_ok=True)
        logpath = log_dir + '/dedupe-staging-' + now + '.log'

    logger.add_handler(pdslogger.file_handler(logpath))
    limits = {'info': -1, 'debug': -1, 'normal': -1}
    cmd_args = sys.argv[1:] if argv is None else argv
    logger.open('dedupe-staging ' + ' '.join(cmd_args), limits=limits)
    return logger


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
        logger.close()

    if any(result.problems for result in results):
        sys.exit(1)


if __name__ == '__main__':
    main()

##########################################################################################
