##########################################################################################
# queue_manager/config.py
#
# This file contains configuration & variables used for task queue & queue mananger.
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

# max allowed suborprocess time in seconds
MAX_ALLOWED_TIME = 60 * 30
# max number of subprocesses to run at a time for one hst pipeline on a proposal id
MAX_SUBPROCESS_CNT = 10

# Task to script command mapping. {P} will be replaced by proposal id and {V} will be
# replaced by a two character visit or multiple visits separated by spaces (for
# pipeline_update_hst_program).
TASK_NUM_TO_CMD_MAPPING = {
    0: 'HST/pipeline/pipeline_query_hst_moving_targets.py --prog-id {P} --tq',
    1: 'HST/pipeline/pipeline_query_hst_products.py --prog-id {P} --tq',
    2: 'HST/pipeline/pipeline_update_hst_program.py --prog-id {P} --vi {V}',
    3: 'HST/pipeline/pipeline_get_program_info.py --prog-id {P}',
    4: 'HST/pipeline/pipeline_update_hst_visit.py --prog-id {P} --vi {V}',
    5: 'HST/pipeline/pipeline_retrieve_hst_visit.py --prog-id {P} --vi {V}',
    6: 'HST/pipeline/pipeline_label_hst_products.py --prog-id {P} --vi {V}',
    7: 'HST/pipeline/pipeline_prepare_browse_products.py --prog-id {P} --vi {V}',
    8: 'HST/pipeline/pipeline_finalize_hst_bundle.py --prog-id {P}',
}

# Task to prority mapping. The larger the number, the higher the priority.
TASK_NUM_TO_PRI_MAPPING = {
    0: 1,
    1: 1,
    2: 2,
    3: 5,
    4: 3,
    5: 4,
    6: 5,
    7: 5,
    8: 5
}

# Current task to previous task mapping
TASK_NUM_TO_PREV_TASK_MAPPING = {
    0: None,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 4,
}
