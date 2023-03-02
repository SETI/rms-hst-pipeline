##########################################################################################
# finalize_hst_bundle/__init__.py
##########################################################################################
import datetime
import os
import pdslogger


from finalize_document import label_hst_document_directory
from finalize_schema import label_hst_schema_directory
from finalize_context import label_hst_context_directory
from finalize_data_product import label_hst_data_directory
from organize_files import organize_files_from_staging_to_bundles

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
    label_hst_document_directory(proposal_id, logger)
    # Generate the final schema directory
    label_hst_schema_directory(proposal_id, logger)
    # Generate the final context directory
    label_hst_context_directory(proposal_id, logger)
    # Organize files, move from staging to bundles
    organize_files_from_staging_to_bundles(proposal_id, logger)
    # Move and create data collection files
    label_hst_data_directory(proposal_id, logger)
