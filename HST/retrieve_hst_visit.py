##########################################################################################
# retrieve_hst_visit.py
#
# retrieve_hst_visit is the main function called in retrieve_hst_visit pipeline task
# script. It will download all identified files from MAST to
# <HST_STAGING>/hst_<nnnnn>/visit_<ss>/ directory.
##########################################################################################

import os
import pdslogger
import shutil

from hst_helper import TRL_CHECKSUMS_FILE
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path)
from hst_helper.query_utils import (download_files,
                                    get_filtered_products,
                                    query_mast_slice)
from queue_manager.task_queue_db import remove_all_tasks_for_a_prog_id

def retrieve_hst_visit(proposal_id, visit, logger=None, testing=False):
    """Retrieve all accepted files for a given proposal id & visit.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.
        logger         pdslogger to use; None for default EasyLogger.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.

    Returns:    the number of files downloaded.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Retrieve hst visit with proposal id: {proposal_id} & visit: {visit}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Query MAST
    table = query_mast_slice(proposal_id=proposal_id, logger=logger)
    filtered_products = get_filtered_products(table, visit)
    files_dir = get_program_dir_path(proposal_id, visit, 'staging', testing)

    try:
        # Download all accepted files
        download_files(filtered_products, files_dir, logger, testing)
    except:
        # Downloading failed, removed the visit folder under the staging directory, and
        # the trl file under pipeline directory. We will only have either all files
        # downloaded or zero file downloaded.
        shutil.rmtree(files_dir)
        os.remove(f'{get_program_dir_path(proposal_id, visit)}/{TRL_CHECKSUMS_FILE}')

        # Before raising the error, remove the task queue of the proposal id from
        # database.
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)
        remove_all_tasks_for_a_prog_id(formatted_proposal_id)
        logger.exception('MAST trl files downlaod failure')
        raise

    return len(filtered_products)
