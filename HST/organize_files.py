##########################################################################################
# organize_files.py
#
# Move all data product files for a given proposal id from the staging folder to the
# bundle folder.
##########################################################################################

import os
import pdslogger
import shutil

from hst_helper import (COL_NAME_PREFIX,
                        MAST_DOWNLOAD_DIRNAME)
from hst_helper.fs_utils import (get_program_dir_path,
                                 get_deliverable_path)

def organize_files_from_staging_to_bundles(proposal_id, logger):
    """Move files from staging folder to bundles folder

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Organize files for proposal id: {proposal_id}')
    # TODO: change copying files to moving files?
    # 1. Move existing files based on PDS4-VERSIONING.txt (need to get this file)
    # 2. Walk through all the downloaded files from MAST in the staging folder and move
    # them over to the bundles folder
    staging_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
    deliverable_path = get_deliverable_path(proposal_id)
    for dir in os.listdir(staging_dir):
        for col_prefix in COL_NAME_PREFIX:
            if dir.startswith(col_prefix):
                staging_prod_dir = os.path.join(staging_dir, dir)
                bundles_prod_dir = os.path.join(deliverable_path, dir)
                os.makedirs(bundles_prod_dir, exist_ok=True)
                logger.info(f'Move {dir} from staging to bundles directory')
                shutil.copytree(staging_prod_dir, bundles_prod_dir, dirs_exist_ok=True)

def clean_up_staging_dir(proposal_id, logger):
    """Remove organized directories (they are copied to the bundle directory) and empty
    mastDownload directories (empty directories) from the staging folder. This step should
    be run at the end of the pipeline process.

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Clean up staging directory for proposal id: {proposal_id}')
    staging_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
    for dir in os.listdir(staging_dir):
        staging_prod_dir = os.path.join(staging_dir, dir)
        # Clean up the empty mastDownload directory
        if dir.startswith(MAST_DOWNLOAD_DIRNAME):
            logger.info(f'Remove {dir} from staging directory')
            shutil.rmtree(staging_prod_dir)
        for col_prefix in COL_NAME_PREFIX:
            if dir.startswith(col_prefix):
                logger.info(f'Remove {dir} from staging directory')
                shutil.rmtree(staging_prod_dir)
