##########################################################################################
# finalize_hst_bundle/__init__.py
##########################################################################################
import datetime
import os
import pdslogger
import shutil

from finalize_document import label_hst_document_directory
from finalize_schema import label_hst_schema_directory
from finalize_context import label_hst_context_directory

from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path,
                                 get_instrument_id_from_fname,
                                 get_file_suffix)

def finalize_hst_bundle(proposal_id, logger=None):
    """With a given proposal id, finalize hst bundle.
    1. Create documents/schema/context/kernel directories.
    2. TODO: Move existing/superseded files based on PDS4-VERSIONING.txt.
    3. Move new files into the proper directories under <HST_BUNDLES>/hst_<nnnnn>/.
    4. Create the new collection.csv and bundle.xml files
    5. Run the validator.

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Finalize hst bundle with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Generate the final document directory
    # label_hst_document_directory(proposal_id, logger)
    # Generate the final schema directory
    # label_hst_schema_directory(proposal_id, logger)
    # Generate the final context directory
    # label_hst_context_directory(proposal_id, logger)

    # TODO: change copying files to moving files?
    # 1. Move existing files based on PDS4-VERSIONING.txt (need to get this file)
    # 2. Walk through all the downloaded files from MAST in the staging folder and move
    # them over to the bundles folder
    staging_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    for dir in os.listdir(staging_dir):
        if dir.startswith('data_') or dir.startswith('browse_'):
            staging_prod_dir = os.path.join(staging_dir, dir)
            bundles_prod_dir = os.path.join(bundles_dir, dir)
            os.makedirs(bundles_prod_dir, exist_ok=True)
            logger.info(f'Move {dir} from staging to bundles directory')
            shutil.copytree(staging_prod_dir, bundles_prod_dir, dirs_exist_ok=True)
#
