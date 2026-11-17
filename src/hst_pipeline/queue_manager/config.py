##########################################################################################
# queue_manager/config.py
#
# This file contains configuration & variables used for task queue & queue manager.
##########################################################################################

import sys

from hst_pipeline.hst_helper import HST_DIR

# python executable
PYTHON_EXE = sys.executable

# task queue db
DB_PATH = f'{HST_DIR["pipeline"]}/task_queue.db'
DB_URI = f'sqlite:///{DB_PATH}'

# max allowed subprocess time in seconds, downloading may take hours.
# Default to 1 day
MAX_ALLOWED_TIME = 60 * 60 * 24

# max number of subprocesses allowed to run at the same time for the pipeline process for all ids.
# Default to 100
MAX_SUBPROCESS_CNT = 100
SUBPROCESS_LIST = []

# Seconds to wait after pipeline finish before re-queuing query_moving_targ
# when run_forever is enabled and the task queue is empty.
# Default to 30 days
REQUEUE_TIME = 60 * 60 * 24 * 30

# Seconds between idle heartbeat messages while waiting for the next re-queue.
# Default to 10 mins.
HEARTBEAT_INTERVAL = 60 * 10

# Seconds to wait after queuing finalize_bundle before it becomes eligible to run.
# Default to 1 days
FINALIZE_BUNDLE_DELAY_TIME = 60 * 60 * 24

# Tasks dropped from the queue database on restart (re-entered via higher-level steps).
LOWER_LVL_TASKS = (
    'get_prog_info',
    'update_visit',
    'retrieve_visit',
    'label_prod',
    'prep_browse_prod',
    'finalize_bundle',
)

# tuple contains (task order, task priority, task command, wrapper-task flag,
# execution delay in seconds).
# task order: the executing order of a task when running pipeline with a proposal id.
# task priority: the larger the number, the higher the priority.
# task command: the console script for each task, as declared in [project.scripts] in
#               pyproject.toml. {P} will be replaced by proposal id and
#               {V} will be replaced by a two character visit or multiple visits separated
#               by spaces (for pipeline_update_hst_program).
# wrapper task: True/False to indicate whether the task is a wrapper that includes multiple
#               subtasks.
# execution delay: seconds after queuing before the task becomes eligible to run; 0 means
#                  no delay (execution_time is null).
TASK_INFO = {
    # get a list of proposal ids based on the search constraints
    'query_moving_targ': (
        0, 1,
        'hst-query-moving-targets --proposal-ids {P} --tq',
        False,
        0,
    ),
    # For each proposal id, we will run the rest of the tasks below:
    # get a list of visits with new/changed files for a given proposal id
    'query_prod': (
        1, 1,
        'hst-query-products --proposal-id {P} --tq',
        False,
        0,
    ),
    # a wrapper to perform these tasks for a given proposal id:
    # - get program info for the proposal id
    # for each of its visits:
    # - update visit for all visits
    # - finalize the bundle for the give proposal id & visits
    'update_prog': (
        2, 2,
        'hst-update-program --proposal-id {P} --visits {V}',
        True,
        0,
    ),
    # download the proposal files (apt/pdf/pro/props) for a given proposal id
    'get_prog_info': (
        3, 5,
        'hst-get-program-info --proposal-id {P}',
        False,
        0,
    ),
    # a wrapper to perform these tasks for a given proposal id & a visit
    # - retrieve files from MAST for a given proposal id & a visit
    # - label each product files
    # - prepare browse products by organizing the downloaded files from MAST
    'update_visit': (
        4, 3,
        'hst-update-visit --proposal-id {P} --vi {V}',
        True,
        0,
    ),
    # download files from MAST for a given proposal id & a visit
    'retrieve_visit': (
        5, 4,
        'hst-retrieve-visit --proposal-id {P} --vi {V}',
        False,
        0,
    ),
    # generate the labels for downloaded files from MAST
    'label_prod': (
        6, 5,
        'hst-label-products --proposal-id {P} --vi {V}',
        False,
        0,
    ),
    # organize downloaded files from MAST, move browse/data products to corresponding directories
    'prep_browse_prod': (
        7, 5,
        'hst-prepare-browse --proposal-id {P} --vi {V}',
        False,
        0,
    ),
    # prepare the final deliverable bundle for a given proposal id after all visits processed
    'finalize_bundle': (
        8, 6,
        'hst-finalize-bundle --proposal-id {P}',
        False,
        FINALIZE_BUNDLE_DELAY_TIME,
    )
}
