##########################################################################################
# finalize_product/__init__.py
##########################################################################################
import datetime
import os
import pdslogger
import shutil

from product_labels.suffix_info import (INSTRUMENT_NAMES,
                                        collection_name,
                                        get_processing_level,
                                        get_collection_title_fmt)
from product_labels.wavelength_ranges   import wavelength_ranges

from hst_helper import COL_NAME_PREFIX
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_format_term,
                                 get_program_dir_path)
from hst_helper.general_utils import (create_xml_label,
                                      create_collection_label,
                                      create_csv,
                                      get_inst_params_from_label,
                                      get_citation_info,
                                      get_instrument_id_set,
                                      get_mod_history_from_label,
                                      get_primary_res_from_label,
                                      get_rec_num_from_label,
                                      get_target_id_from_label)

COL_DATA_LABEL_TEMPLATE = 'PRODUCT_COLLECTION_LABEL.xml'

def label_hst_data_directory(proposal_id, logger):
    """With a given proposal id, move data directory in the final bundle.
    1. Move data directory from staging to bundles directory.
    2. Create data csv.
    3. create data xml label.

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst data directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Move files
    move_files_from_staging_to_bundles(proposal_id, logger)

    # Collect data to construct data dictionary used for the labels
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    for dir in os.listdir(bundles_dir):
        for col_name in COL_NAME_PREFIX:
            if dir.startswith(col_name): # work on data_directory
                col_name = dir

                # Get channel id
                prod_dir = os.path.join(bundles_dir, dir)
                _, channel_id, _, _ = get_inst_params_from_label(prod_dir)

                # Get collection title
                _, inst_id, suffix = dir.split('_')
                collection_title = get_collection_title_fmt(suffix, inst_id.upper())
                collection_title = collection_title.replace('{I}', inst_id.upper())
                collection_title = collection_title.replace(
                                       '{IC}', inst_id.upper() + f'/{channel_id}')
                collection_title = collection_title.replace('{P}', str(proposal_id))

                #  Get primary results
                (_, processing_lvl,
                 wavelength_ranges, _) = get_primary_res_from_label(prod_dir)

                # Get citation info
                citation_info = get_citation_info(proposal_id, logger)

                version_id = (1, 0)
                col_data_label_name = f'collection_{col_name}.xml'
                mod_history = get_mod_history_from_label(col_data_label_name, version_id)

                # get target identification
                # TODO: might need to walk through bundles dir depending on if the files have
                # been moved to the bundles dir.
                # files_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
                col_data_label_path = os.path.join(prod_dir, col_data_label_name)
                target_info = get_target_id_from_label(proposal_id,
                                                       col_data_label_path,
                                                       prod_dir)

                # Get label date
                timetag = os.path.getmtime(__file__)
                label_date = datetime.datetime.fromtimestamp(timetag).strftime("%Y-%m-%d")

                records_num =  get_rec_num_from_label(prod_dir)

                data_dict = {
                    'prop_id': proposal_id,
                    'collection_name': col_name,
                    'collection_title': collection_title,
                    'citation_info': citation_info,
                    'processing_level': processing_lvl[0],
                    'wavelength_ranges': wavelength_ranges,
                    'instrument_name': INSTRUMENT_NAMES[inst_id.upper()],
                    'target_identifications': target_info,
                    'version_id': version_id,
                    'label_date': label_date,
                    'inst_id': inst_id,
                    # 'csv_filename': CSV_FILENAME,
                    'records_num': records_num,
                    'mod_history': mod_history,
                }


                # Create context collection label
                create_collection_label(proposal_id, col_name,
                                        data_dict, col_data_label_name,
                                        COL_DATA_LABEL_TEMPLATE, logger, prod_dir)


    # Create product collection csv
    create_product_collection_csv(proposal_id, None, logger)

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
        for col_name in COL_NAME_PREFIX:
            if dir.startswith(col_name):
                staging_prod_dir = os.path.join(staging_dir, dir)
                bundles_prod_dir = os.path.join(bundles_dir, dir)
                os.makedirs(bundles_prod_dir, exist_ok=True)
                logger.info(f'Move {dir} from staging to bundles directory')
                shutil.copytree(staging_prod_dir, bundles_prod_dir, dirs_exist_ok=True)

def create_product_collection_csv(proposal_id, data_dict, logger):
    """With a given proposal id, create product collection csv in the final bundle.

    Inputs:
        proposal_id:    a proposal id.
        data_dict:      data dictonary to fill in the label template.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    prod_ver = (1,0)
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    for dir in os.listdir(bundles_dir):
        for col_name in COL_NAME_PREFIX:
            if dir.startswith(col_name):
                collection_data = []
                bundles_prod_dir = os.path.join(bundles_dir, dir)
                for root, _, files in os.walk(bundles_prod_dir):
                    for file in files:
                        if not file.startswith('collection_') and file.endswith('.xml'):
                            format_term = get_format_term(file)
                            formatted_proposal_id = get_formatted_proposal_id(proposal_id)
                            prod_lidvid = (f'P,urn:nasa:pds:hst_{formatted_proposal_id}'
                                        + f':{dir}:{format_term}::'
                                        + f'{prod_ver[0]}.{prod_ver[1]}').split(',')
                            if prod_lidvid not in collection_data:
                                collection_data.append(prod_lidvid)
                prod_csv = bundles_prod_dir + f'/collection_{dir}.csv'
                create_csv(prod_csv, collection_data, logger)
