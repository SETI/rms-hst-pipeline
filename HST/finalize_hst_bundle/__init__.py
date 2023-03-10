##########################################################################################
# finalize_hst_bundle/__init__.py
##########################################################################################
import datetime
import os
import pdslogger

from product_labels.suffix_info import INSTRUMENT_NAMES
from hst_helper.fs_utils import get_program_dir_path
from hst_helper.general_utils import (date_time_to_date,
                                      get_citation_info,
                                      get_collection_label_data,
                                      get_instrument_id_set)

from finalize_document import label_hst_document_directory
from finalize_schema import label_hst_schema_directory
from finalize_context import label_hst_context_directory
from finalize_data_product import label_hst_data_directory
from organize_files import organize_files_from_staging_to_bundles
from label_bundle import label_hst_bundle
from run_validation import run_validation

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

    # Get the general label data used in document/schema/context/bundle labels
    data_dict = get_general_label_data(proposal_id, logger)
    # Generate the final document directory
    label_hst_document_directory(proposal_id, data_dict, logger)
    # Generate the final schema directory
    label_hst_schema_directory(proposal_id, data_dict, logger)
    # Generate the final context directory
    label_hst_context_directory(proposal_id, data_dict, logger)
    # Organize files, move from staging to bundles
    organize_files_from_staging_to_bundles(proposal_id, logger)
    # Move and create data collection files
    label_hst_data_directory(proposal_id, logger)
    # Create bundle label
    label_hst_bundle(proposal_id, data_dict, logger)
    # Create manifest files
    run_validation(proposal_id, logger)

def get_general_label_data(proposal_id, logger):
    """Get general label data used in document/schema/context/bundle labels

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    # Get citation info
    citation_info = get_citation_info(proposal_id, logger)
    formatted_title = (citation_info.title
    + ", HST Cycle "
    + str(citation_info.cycle)
    + " Program "
    + str(citation_info.propno)
    + ", "
    + citation_info.publication_year
    + "."
    )

    # Get label date
    timetag = os.path.getmtime(__file__)
    label_date = datetime.datetime.fromtimestamp(timetag).strftime("%Y-%m-%d")

    # Get imstrument id
    inst_ids = get_instrument_id_set(proposal_id, logger)

    # Get target id, time, instrument params, primary results and the number of records
    files_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
    label_data = get_collection_label_data(proposal_id, files_dir, logger)
    target_info = label_data['target']
    _, _, wavelength_ranges, _ = label_data['primary_res']
    min_start, max_stop = label_data['time']
    start_date = date_time_to_date(min_start) if min_start else None
    stop_date = date_time_to_date(max_stop) if max_stop else None

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
