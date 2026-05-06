##########################################################################################
# queue_manager/config.py
#
# This file contains configuration & variables used for task queue & queue manager.
##########################################################################################

import os
import sys

from hst_helper import HST_DIR

# python executable
PYTHON_EXE = sys.executable
# root of pds-hst-pipeline dir
HST_SOURCE_ROOT = os.environ['PDS_HST_PIPELINE']

# task queue db
DB_PATH = f'{HST_DIR["pipeline"]}/task_queue.db'
DB_URI = f'sqlite:///{DB_PATH}'

# max allowed subprocess time in seconds, downloading may take hours
MAX_ALLOWED_TIME = 60 * 60 * 24
# max number of subprocesses allowed to run at the same time for the pipeline process for
# all ids.
MAX_SUBPROCESS_CNT = 20
SUBPROCESS_LIST = []

# A dictionary keyed by task name, and its corresponding task tuple as the value. Each
# tuple contains (task order, task priority, and task command).
# task order: the executing order of a task when running pipeline with a proposal id.
# task priority: the larger the number, the higher the priority.
# task command: the script command for each task. {P} will be replaced by proposal id and
#               {V} will be replaced by a two character visit or multiple visits separated
#               by spaces (for pipeline_update_hst_program).
TASK_INFO = {
    # get a list of proposal ids based on the search constraints
    'query_moving_targ': (
        0, 1,
        'HST/pipeline/pipeline_query_hst_moving_targets.py --proposal-ids {P} --tq'
    ),
    # For each proposal id, we will run the rest of the tasks below:
    # get a list of visits with new/changed files for a given proposal id
    'query_prod': (
        1, 1,
        'HST/pipeline/pipeline_query_hst_products.py --proposal-id {P} --tq'
    ),
    # a wrapper to perform these tasks for a given proposal id:
    # - get program info for the proposal id
    # for each of its visits:
    # - update visit for all visits
    # - finalize the bundle for the give proposal id & visits
    'update_prog': (
        2, 2,
        'HST/pipeline/pipeline_update_hst_program.py --proposal-id {P} --visits {V}'
    ),
    # download the proposal files (apt/pdf/pro/props) for a given proposal id
    'get_prog_info': (
        3, 5,
        'HST/pipeline/pipeline_get_program_info.py --proposal-id {P}'
    ),
    # a wrapper to perform these tasks for a given proposal id & a visit
    # - retrieve files from MAST for a given proposal id & a visit
    # - label each product files
    # - prepare browse products by organizing the downloaded files from MAST
    'update_visit': (
        4, 3,
        'HST/pipeline/pipeline_update_hst_visit.py --proposal-id {P} --vi {V}'
    ),
    # download files from MAST for a given proposal id & a visit
    'retrieve_visit': (
        5, 4,
        'HST/pipeline/pipeline_retrieve_hst_visit.py --proposal-id {P} --vi {V}'
    ),
    # generate the labels for downloaded files from MAST
    'label_prod': (
        6, 5,
        'HST/pipeline/pipeline_label_hst_products.py --proposal-id {P} --vi {V}'
    ),
    # organize downloaded files from MAST, move browse/data products to corresponding directories
    'prep_browse_prod': (
        7, 5,
        'HST/pipeline/pipeline_prepare_browse_products.py --proposal-id {P} --vi {V}'
    ),
    # prepare the final deliverable bundle for a given proposal id after all visits processed
    'finalize_bundle': (
        8, 5,
        'HST/pipeline/pipeline_finalize_hst_bundle.py --proposal-id {P}'
    )
}
