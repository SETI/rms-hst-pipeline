##########################################################################################
# label_bundle/__init__.py
##########################################################################################
import csv
import datetime
import os
import pdslogger
import shutil

from product_labels.suffix_info import (INSTRUMENT_NAMES,
                                        get_collection_title_fmt)
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path)
from hst_helper.general_utils import (create_collection_label,
                                      date_time_to_date,
                                      get_citation_info,
                                      get_collection_label_data,
                                      get_instrument_id_set,
                                      get_mod_history_from_label)

BUNDLE_LABEL = 'bundle.xml'
BUNDLE_LABEL_TEMPLATE = 'BUNDLE_LABEL.xml'

def label_hst_bundle(proposal_id, logger):
    """With a given proposal id, create the bundle label.

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst bundle directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    #  Collect data to construct data dictionary used for the context label
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

    # Get the mod history for bundle label if it's already existed.
    version_id = (1, 0)
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    bundle_label_path = bundles_dir + f'/{BUNDLE_LABEL}'
    mod_history = get_mod_history_from_label(bundle_label_path, version_id)
    # inst_ids = get_instrument_id_set(proposal_id, logger)
    # Get label date
    timetag = os.path.getmtime(__file__)
    label_date = datetime.datetime.fromtimestamp(timetag).strftime("%Y-%m-%d")

    # Get imstrument id
    inst_ids = get_instrument_id_set(proposal_id, logger)

    files_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
    label_data = get_collection_label_data(proposal_id, files_dir, logger)
    target_info = label_data['target']
    _, processing_lvl, wavelength_ranges, _ = label_data['primary_res']
    min_start, max_stop = label_data['time']
    start_date = date_time_to_date(min_start) if min_start else None
    stop_date = date_time_to_date(max_stop) if max_stop else None

    # TODO: determine the version of each entry
    bundle_entries = []
    for col_name in os.listdir(bundles_dir):
        if '.' not in col_name:
            col_type, _, _ = col_name.partition('_')
            col_ver = (1,0)
            bundle_entries.append((col_name, col_type, col_ver))


    data_dict = {
        'prop_id': proposal_id,
        'collection_name': 'bundle',
        'citation_info': citation_info,
        'formatted_title': formatted_title,
        'mod_history': mod_history,
        'version_id': version_id,
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
        'bundle_entry_li': bundle_entries
    }

    # Create bundle collection label
    create_collection_label(proposal_id, 'bundle', data_dict,
                            BUNDLE_LABEL, BUNDLE_LABEL_TEMPLATE, logger)
