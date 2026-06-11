#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_finalize_hst_bundle.py
#
# Syntax:
# pipeline_finalize_hst_bundle.py [-h] --proposal-id PROPOSAL_ID [--log LOG] [--quiet]
#
# Enter the --help option to see more information.
#
# Perform finalize_hst_bundle task to package a complete set of files in the staging
# directories as a new bundle or as updates to an existing bundle. All bundle files
# will be stored in the bundles directories.
##########################################################################################

import argparse
import datetime
import os
import pdslogger
import sys

from finalize_hst_bundle import finalize_hst_bundle
from hst_helper import HST_DIR
from hst_helper.fs_utils import get_formatted_proposal_id
from queue_manager.task_queue_db import remove_a_task


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_finalize_hst_bundle: package a complete set of files in the
                    staging directories as a new bundle or as updates to an existing bundle.
                    """)

    parser.add_argument('--proposal-id', type=str, default='', required=True,
        help='The proposal id for the MAST query.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "finalize-hst-bundle-<date>.log".""")

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

    logger = pdslogger.PdsLogger('pds.hst.finalize-hst-bundle-' + proposal_id)
    if not args.quiet:
        logger.add_handler(pdslogger.stdout_handler)

    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if args.log:
        if os.path.isdir(args.log):
            logpath = os.path.join(args.log, 'finalize-hst-bundle-' + now + '.log')
        else:
            parts = os.path.splitext(args.log)
            logpath = parts[0] + '-' + now + parts[1]
    else:
        os.makedirs(log_dir, exist_ok=True)
        logpath = log_dir + '/finalize-hst-bundle-' + now + '.log'

    logger.add_handler(pdslogger.file_handler(logpath))
    limits = {'info': -1, 'debug': -1, 'normal': -1}
    cmd_args = sys.argv[1:] if argv is None else argv
    logger.open('finalize-hst-bundle ' + ' '.join(cmd_args), limits=limits)
    return logger


def main(argv=None):
    """Perform the finalize_hst_bundle pipeline task."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        proposal_id = args.proposal_id
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)

        try:
            finalize_hst_bundle(proposal_id, logger)
        except Exception as e:
            logger.exception(e)
            raise

        remove_a_task(formatted_proposal_id, '', 'finalize_bundle')
    finally:
        logger.close()


if __name__ == '__main__':
    main()

##########################################################################################
