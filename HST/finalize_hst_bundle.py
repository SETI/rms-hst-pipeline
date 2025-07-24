##########################################################################################
# finalize_hst_bundle.py
#
# finalize_hst_bundle is the main function called in finalize_hst_bundle pipeline task
# script. It will do these actions:
#
# - Create documents/schema/context/kernel directories.
# - Move new files into the proper directories under <HST_BUNDLES>/hst_<nnnnn>/.
# - Create the new collection csv & xml and the bundle xml files
# - Run the validator.
##########################################################################################

import datetime
import os
import pdslogger

from create_target_label import create_target_label
from finalize_document import finalize_hst_document_directory
from finalize_schema import finalize_hst_schema_directory
from finalize_context import finalize_hst_context_directory
from finalize_data_product import finalize_hst_data_directory
from hst_helper.fs_utils import get_program_dir_path
from hst_helper.general_utils import (date_time_to_date,
                                      get_citation_info,
                                      get_collection_label_data,
                                      get_instrument_id_set)
from label_bundle import label_hst_bundle
from organize_files import organize_files_from_staging_to_bundles
from product_labels.suffix_info import INSTRUMENT_NAMES
from run_validation import run_validation

from hst_helper import CITATION_INFO_DICT

def finalize_hst_bundle(proposal_id, logger=None):
    """With a given proposal id, finalize hst bundle.

    1. Create documents/schema/context/kernel directories.
    2. TODO: Move existing/superseded files based on PDS4-VERSIONING.txt.
    3. Move new files into the proper directories under <HST_BUNDLES>/hst_<nnnnn>/.
    4. Create the new collection.csv and bundle.xml files
    5. Run the validator.

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Finalize hst bundle with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Get the general label data used in document/schema/context/bundle labels
    data_dict = get_general_label_data(proposal_id, logger)
    # Generate the final document directory
    finalize_hst_document_directory(proposal_id, data_dict, logger)
    # Generate the final schema directory
    finalize_hst_schema_directory(proposal_id, data_dict, logger)
    # Generate the final context directory
    finalize_hst_context_directory(proposal_id, data_dict, logger)
    # Organize files, move from staging to bundles
    organize_files_from_staging_to_bundles(proposal_id, logger)
    # Create data collection files
    finalize_hst_data_directory(proposal_id, logger)
    # Create bundle label
    label_hst_bundle(proposal_id, data_dict, logger)
    # Create target label if it doesn't exist in PDS page
    create_target_label(proposal_id, data_dict, logger)
    # Create manifest files & run validator
    run_validation(proposal_id, logger)

def get_general_label_data(proposal_id, logger=None, testing=False):
    """Get general label data used in document/schema/context/bundle labels

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.

    Returns:    a dictionary of general data that will be used for the creation of
                document/schema/context/bundle labels.
    """
    logger = logger or pdslogger.EasyLogger()

    # Get citation info
    citation_info = get_citation_info(proposal_id, logger) if not testing else None
    formatted_title = (f'{citation_info.title}, HST Cycle {citation_info.cycle} Program '
                       f'{citation_info.propno}, {citation_info.publication_year}.')

    # Get label date
    timetag = os.path.getmtime(__file__)
    label_date = datetime.datetime.fromtimestamp(timetag).strftime('%Y-%m-%d')

    # Get instrument id
    inst_ids = get_instrument_id_set(proposal_id, logger)

    # Get target id, time, instrument params, primary results and the number of records
    files_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
    label_data = get_collection_label_data(proposal_id, files_dir, logger)
    target_info = label_data['target']
    _, _, wavelength_ranges, _ = label_data['primary_res']
    min_start, max_stop = label_data['time']
    start_date = date_time_to_date(min_start) if min_start else None
    stop_date = date_time_to_date(max_stop) if max_stop else None

    # get clean text for target lidvid
    for targ in target_info:
        lid = targ['lid']
        lid_li = lid.split('.')
        targ_type = lid_li[0]
        targ_name = lid_li[-1]
        targ['formatted_name'] = targ_name
        targ['formatted_type'] = targ_type
        targ['lid'] = f'urn:nasa:pds:context:target:{lid}'

    data_dict = {
        'prop_id': proposal_id,
        'collection_name': 'bundle',
        'citation_info': citation_info,
        'formatted_title': formatted_title,
        'processing_level': 'Raw',
        'wavelength_ranges': wavelength_ranges,
        'instrument_name_dict': INSTRUMENT_NAMES,
        'target_identifications': target_info,
        'label_date': label_date,
        'inst_id_li': list(inst_ids),
        'start_date_time': min_start,
        'stop_date_time': max_stop,
        'start_date': start_date,
        'stop_date': stop_date,
    }

    return data_dict
