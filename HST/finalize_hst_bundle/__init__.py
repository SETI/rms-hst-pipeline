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
                                 get_format_term,
                                 get_program_dir_path,
                                 get_instrument_id_from_fname,
                                 get_file_suffix)
from hst_helper.general_utils import (create_xml_label,
                                      create_collection_label,
                                      create_csv,
                                      get_citation_info,
                                      get_instrument_id_set,
                                      get_mod_history_from_label)

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
    # Move files
    # move_files_from_staging_to_bundles(proposal_id, logger)

    prod_ver = (1,0)
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    for dir in os.listdir(bundles_dir):
        if dir.startswith('data_') or dir.startswith('browse_'):
            collection_data = []
            bundles_prod_dir = os.path.join(bundles_dir, dir)
            for root, _, files in os.walk(bundles_prod_dir):
                for file in files:
                    if file.endswith('.xml'):
                        format_term = get_format_term(file)
                        formatted_proposal_id = get_formatted_proposal_id(proposal_id)
                        prod_lidvid = (f'P,urn:nasa:pds:hst_{formatted_proposal_id}'
                                    + f':{dir}:{format_term}::'
                                    + f'{prod_ver[0]}.{prod_ver[1]}').split(',')
                        if prod_lidvid not in collection_data:
                            collection_data.append(prod_lidvid)
            prod_csv_dir = bundles_prod_dir + f'/collection_{dir}.csv'
            create_csv(prod_csv_dir, collection_data, logger)

def move_files_from_staging_to_bundles(proposal_id, logger):
    """Move files from staging folder to bundles folder

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
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
