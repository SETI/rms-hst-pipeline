##########################################################################################
# finalize_data_product.py
#
# - Move data directory from staging to bundles directory.
# - Create data csv.
# - Create data xml label.
##########################################################################################

import datetime
import os
import pdslogger

from hst_helper import (COL_NAME_PREFIX,
                        BROWSE_PROD_EXT)
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_format_term,
                                 get_deliverable_path)
from hst_helper.general_utils import (create_collection_label,
                                      create_csv,
                                      date_time_to_date,
                                      get_citation_info,
                                      get_collection_label_data,
                                      get_mod_history_from_label)
from product_labels.suffix_info import (INSTRUMENT_NAMES,
                                        get_collection_title_fmt)

COL_DATA_LABEL_TEMPLATE = 'PRODUCT_COLLECTION_LABEL.xml'

def label_hst_data_directory(proposal_id, logger):
    """With a given proposal id, move data directory in the final bundle.

    1. Move data directory from staging to bundles directory.
    2. Create data csv.
    3. create data xml label.

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst data directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Collect data to construct data dictionary used for the labels
    deliverable_path = get_deliverable_path(proposal_id)
    for dir in os.listdir(deliverable_path):
        for col_prefix in COL_NAME_PREFIX:
            if dir.startswith(col_prefix): # work on data_directory
                collection_name = dir

                # Get channel id
                prod_dir = os.path.join(deliverable_path, dir)

                # Get label data
                # TODO: might need to walk through bundles dir depending on if the files
                # have been moved to the bundles dir.
                try:
                    label_data = get_collection_label_data(proposal_id, prod_dir, logger)
                    target_info = label_data['target']

                    # For browse products, we don't have primary_res & inst_params since
                    # those products don't have data .xml label
                    try:
                        _, proc_lvl, wavelength_ranges, _ = label_data['primary_res']
                        processing_lvl = proc_lvl[0]
                    except KeyError:
                        processing_lvl = None
                        wavelength_ranges = None
                    try:
                        _, channel_id, _, _ = label_data['inst_params']
                    except KeyError:
                        channel_id = None

                    records_num = label_data['records']
                    min_start, max_stop = label_data['time']
                    start_date = date_time_to_date(min_start) if min_start else None
                    stop_date = date_time_to_date(max_stop) if max_stop else None

                    # Get collection title
                    _, inst_id, suffix = dir.split('_')
                    collection_title = get_collection_title_fmt(suffix, inst_id.upper())
                    collection_title = collection_title.replace('{I}', inst_id.upper())
                    collection_title = collection_title.replace(
                                        '{IC}', f'{inst_id.upper()}/{channel_id}')
                    collection_title = collection_title.replace('{P}', str(proposal_id))

                    # Get citation info
                    citation_info = get_citation_info(proposal_id, logger)

                    version_id = (1, 0)
                    col_data_label_name = f'collection_{collection_name}.xml'
                    mod_history = get_mod_history_from_label(col_data_label_name,
                                                             version_id)

                    # Get label date
                    timetag = os.path.getmtime(__file__)
                    label_date = (datetime.datetime.fromtimestamp(timetag)
                                                   .strftime('%Y-%m-%d'))
                except Exception as e:
                    logger.exception(e)
                    raise
                data_dict = {
                    'prop_id': proposal_id,
                    'inst_id': inst_id,
                    'collection_name': collection_name,
                    'collection_title': collection_title,
                    'citation_info': citation_info,
                    'processing_level': processing_lvl,
                    'wavelength_ranges': wavelength_ranges,
                    'instrument_name': INSTRUMENT_NAMES[inst_id.upper()],
                    'target_identifications': target_info,
                    'version_id': version_id,
                    'label_date': label_date,
                    'records_num': records_num,
                    'mod_history': mod_history,
                    'start_date_time': min_start,
                    'stop_date_time': max_stop,
                    'start_date': start_date,
                    'stop_date': stop_date
                }

                # Create data product collection label
                create_collection_label(proposal_id, collection_name,
                                        data_dict, col_data_label_name,
                                        COL_DATA_LABEL_TEMPLATE, logger)

    # Create data product collection csv
    create_data_product_collection_csv(proposal_id, logger)

def create_data_product_collection_csv(proposal_id, logger):
    """With a given proposal id, create data product collection csv in the final bundle.

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.
    """
    prod_ver = (1,0)
    deliverable_path = get_deliverable_path(proposal_id)
    for dir in os.listdir(deliverable_path):
        for col_prefix in COL_NAME_PREFIX:
            if dir.startswith(col_prefix):
                collection_data = []
                bundles_prod_dir = os.path.join(deliverable_path, dir)
                for _, _, files in os.walk(bundles_prod_dir):
                    for file in files:
                        _, _, ext = file.rpartition('.')
                        # if not file.startswith('collection_') and file.endswith('.xml'):
                        if (not file.startswith('collection_') and
                            ext in ['xml'] + BROWSE_PROD_EXT):
                            format_term = get_format_term(file)
                            formatted_proposal_id = get_formatted_proposal_id(proposal_id)
                            prod_lidvid = (f'P,urn:nasa:pds:hst_{formatted_proposal_id}'
                                           f':{dir}:{format_term}::'
                                           f'{prod_ver[0]}.{prod_ver[1]}').split(',')
                            if prod_lidvid not in collection_data:
                                collection_data.append(prod_lidvid)
                prod_csv = f'{bundles_prod_dir}/collection_{dir}.csv'
                create_csv(prod_csv, collection_data, logger)
