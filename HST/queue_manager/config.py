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

# max allowed subprocess time in seconds
MAX_ALLOWED_TIME = 60 * 30
# max number of subprocesses to run at a time for one hst pipeline on a proposal id
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
    'query_moving_targ': (
        0, 1, 'HST/pipeline/pipeline_query_hst_moving_targets.py --prog-id {P} --tq'
    ),
    'query_prod': (
        1, 1, 'HST/pipeline/pipeline_query_hst_products.py --prog-id {P} --tq'
    ),
    'update_prog': (
        2, 2, 'HST/pipeline/pipeline_update_hst_program.py --prog-id {P} --vi {V}'
    ),
    'get_prog_info': (
        3, 5, 'HST/pipeline/pipeline_get_program_info.py --prog-id {P}'
    ),
    'update_visit': (
        4, 3, 'HST/pipeline/pipeline_update_hst_visit.py --prog-id {P} --vi {V}'
    ),
    'retrieve_visit': (
        5, 4, 'HST/pipeline/pipeline_retrieve_hst_visit.py --prog-id {P} --vi {V}'
    ),
    'label_prod': (
        6, 5, 'HST/pipeline/pipeline_label_hst_products.py --prog-id {P} --vi {V}'
    ),
    'prep_browse_prod': (
        7, 5, 'HST/pipeline/pipeline_prepare_browse_products.py --prog-id {P} --vi {V}'
    ),
    'finalize_bundle': (
        8, 5, 'HST/pipeline/pipeline_finalize_hst_bundle.py --prog-id {P}'
    )
}

# # Task to script command mapping. {P} will be replaced by proposal id and {V} will be
# # replaced by a two character visit or multiple visits separated by spaces (for
# # pipeline_update_hst_program).
# TASK_TO_CMD_MAPPING = {
#     'query_moving_targ': 'HST/pipeline/pipeline_query_hst_moving_targets.py' +
#                          ' --prog-id {P} --tq',
#     'query_prod': 'HST/pipeline/pipeline_query_hst_products.py --prog-id {P} --tq',
#     'update_prog': 'HST/pipeline/pipeline_update_hst_program.py --prog-id {P} --vi {V}',
#     'get_prog_info': 'HST/pipeline/pipeline_get_program_info.py --prog-id {P}',
#     'update_visit': 'HST/pipeline/pipeline_update_hst_visit.py --prog-id {P} --vi {V}',
#     'retrieve_visit': 'HST/pipeline/pipeline_retrieve_hst_visit.py' +
#                       ' --prog-id {P} --vi {V}',
#     'label_prod': 'HST/pipeline/pipeline_label_hst_products.py --prog-id {P} --vi {V}',
#     'prep_browse_prod': 'HST/pipeline/pipeline_prepare_browse_products.py' +
#                         ' --prog-id {P} --vi {V}',
#     'finalize_bundle': 'HST/pipeline/pipeline_finalize_hst_bundle.py --prog-id {P}',
# }

# TASK_ORDER = {
#     'query_moving_targ': 0,
#     'query_prod': 1,
#     'update_prog': 2,
#     'get_prog_info': 3,
#     'update_visit': 4,
#     'retrieve_visit': 5,
#     'label_prod': 6,
#     'prep_browse_prod': 7,
#     'finalize_bundle': 8
# }

# # Task to prority mapping. The larger the number, the higher the priority.
# TASK_TO_PRI_MAPPING = {
#     'query_moving_targ': 1,
#     'query_prod': 1,
#     'update_prog': 2,
#     'get_prog_info': 5,
#     'update_visit': 3,
#     'retrieve_visit': 4,
#     'label_prod': 5,
#     'prep_browse_prod': 5,
#     'finalize_bundle': 5
# }

# # Current task to previous task mapping
# TASK_TO_PREV_TASK_MAPPING = {
#     'query_moving_targ': None,
#     'query_prod': 'query_moving_targ',
#     'update_prog': 'query_prod',
#     'get_prog_info': 'update_prog',
#     'update_visit': 'get_prog_info',
#     'retrieve_visit': 'update_visit',
#     'label_prod': 'retrieve_visit',
#     'prep_browse_prod': 'label_prod',
#     'finalize_bundle': 'update_visit',
# }
