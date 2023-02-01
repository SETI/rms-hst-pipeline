##########################################################################################
# finalize_hst_bundle/__init__.py
##########################################################################################
import datetime
import os
import pdslogger
import shutil

# from hst_helper import (DOCUMENT_EXT,
#                         PROGRAM_INFO_FILE)

# from hst_helper.fs_utils import (get_program_dir_path,
#                                  get_instrument_id,
#                                  get_file_suffix)

# from citations import Citation_Information
# from xmltemplate import XmlTemplate
from document_label import label_hst_document_directory

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

    label_hst_document_directory(proposal_id, logger)


    return
