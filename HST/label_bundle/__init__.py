##########################################################################################
# label_bundle/__init__.py
##########################################################################################
import os
import pdslogger

from hst_helper.fs_utils import get_program_dir_path
from hst_helper.general_utils import (create_collection_label,
                                      get_mod_history_from_label)

BUNDLE_LABEL = 'bundle.xml'
BUNDLE_LABEL_TEMPLATE = 'BUNDLE_LABEL.xml'

def label_hst_bundle(proposal_id, data_dict, logger):
    """With a given proposal id, create the bundle label.

    Inputs:
        proposal_id:    a proposal id.
        data_dict:      a data dictionary used to create the label.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst bundle directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Get the mod history for bundle label if it's already existed.
    version_id = (1, 0)
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    bundle_label_path = bundles_dir + f'/{BUNDLE_LABEL}'
    mod_history = get_mod_history_from_label(bundle_label_path, version_id)

    # TODO: determine the version of each entry
    bundle_entries = []
    for col_name in os.listdir(bundles_dir):
        if '.' not in col_name:
            col_type, _, _ = col_name.partition('_')
            col_ver = (1,0)
            bundle_entries.append((col_name, col_type, col_ver))

    bundle_data_dict = {
        'collection_name': 'bundle',
        'mod_history': mod_history,
        'version_id': version_id,
        'processing_level': 'Raw',
        'bundle_entry_li': bundle_entries
    }
    bundle_data_dict = {**bundle_data_dict, **data_dict}

    # Create bundle collection label
    create_collection_label(proposal_id, 'bundle', bundle_data_dict,
                            BUNDLE_LABEL, BUNDLE_LABEL_TEMPLATE, logger)
