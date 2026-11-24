#!/usr/bin/env python3
##########################################################################################
# pipeline/pipeline_query_hst_moving_targets.py
#
# Syntax:
# pipeline_query_hst_moving_targets.py [-h] [--proposal-ids PROPOSAL_IDS
#                                            [PROPOSAL_IDS ...]]
#                                      [--instruments INSTRUMENTS [INSTRUMENTS ...]]
#                                      [--start START] [--end END]
#                                      [--retry RETRY] [--log LOG] [--quiet] [--taskqueue]
#
# Enter the --help option to see more information.
#
# Perform query_hst_moving_targets task with these actions:
#
# - Return a list of proposal ids with moving targets based on the query constraints.
# - Queue query_hst_products task if HST_PIPELINE/hst_<nnnnn> directory is missing.
##########################################################################################

import argparse
import sys

from hst_pipeline.hst_helper import (END_DATE,
                                     RETRY,
                                     START_DATE)
from hst_pipeline.hst_helper.fs_utils import get_formatted_proposal_id
from hst_pipeline.hst_helper.log_support import (close_task_logger,
                                                 setup_task_logger)
from hst_pipeline.query_hst_moving_targets import query_hst_moving_targets
from hst_pipeline.queue_manager import queue_next_task
from hst_pipeline.queue_manager.task_queue_db import remove_a_task


def parse_args(argv=None):
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""pipeline_query_hst_moving_targets: Perform MAST query with given query
                    constraints.""")

    parser.add_argument('--proposal-ids', nargs='*', type=str, default=[],
        help='The proposal ids for the MAST query (omit values to query all).')

    parser.add_argument('--instruments', '-i', nargs='+', type=str, default='',
        help='The instruments for the MAST query.')

    parser.add_argument('--start', type=str, action='store', default='',
        help='Optional start date from MAST in (yyyy, mm, dd) format.')

    parser.add_argument('--end', type=str, action='store', default='',
        help='Optional end date from MAST in (yyyy, mm, dd) format.')

    parser.add_argument('--retry', '-r', type=int, default=None,
        help='Optional max number of MAST connection retry.')

    parser.add_argument('--log', '-l', type=str, default='',
        help="""Path and name for the log file. The name always has the current date and time
             appended. If not specified, the file will be written to the current logs
             directory and named "query-hst-moving-targets-<date>.log".""")

    parser.add_argument('--quiet', '-q', action='store_true',
        help='Do not also log to the terminal.')

    parser.add_argument('--taskqueue', '--tq', action='store_true',
        help='Run the script with task queue.')

    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv)


def normalize_proposal_ids(proposal_ids):
    """Return a cleaned proposal id list (empty list means query all ids)."""
    return [p.strip() for p in proposal_ids if p.strip()]


def setup_logger(args, argv=None):
    """Configure and open the pipeline logger; see setup_task_logger()."""
    return setup_task_logger('query-hst-moving-targets', args, argv)


def queue_follow_up_tasks(proposal_ids, program_ids_list, logger):
    """Queue query_hst_products tasks and remove query_moving_targ tasks."""
    for proposal_id in program_ids_list:
        logger.info(f'Queue query_hst_products for {proposal_id}')
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)
        queue_next_task(formatted_proposal_id, '', 'query_prod', logger)

    # Remove per-proposal tasks for every user-requested id, including those
    # that returned no moving targets.
    if proposal_ids:
        for proposal_id in proposal_ids:
            remove_a_task(get_formatted_proposal_id(proposal_id), '', 'query_moving_targ')
    else:
        remove_a_task('', '', 'query_moving_targ')


def main(argv=None):
    """Perform the query_hst_moving_targets pipeline task."""
    args = parse_args(argv)
    logger = setup_logger(args, argv=argv)
    try:
        proposal_ids = normalize_proposal_ids(args.proposal_ids)
        instruments = args.instruments if args.instruments else []
        start_date = args.start if args.start else START_DATE
        end_date = args.end if args.end else END_DATE
        retry = args.retry if args.retry is not None else RETRY

        logger.info('MAST query constraints: ' + str(args))
        program_ids_list = query_hst_moving_targets(proposal_ids=proposal_ids,
                                                    instruments=instruments,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    logger=logger,
                                                    max_retries=retry)
        logger.info('List of program ids: ' + str(program_ids_list))

        if args.taskqueue:
            queue_follow_up_tasks(proposal_ids, program_ids_list, logger)
    finally:
        close_task_logger(logger)


if __name__ == '__main__':
    main()

##########################################################################################
