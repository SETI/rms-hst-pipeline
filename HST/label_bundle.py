##########################################################################################
# label_bundle.py
#
# Create bundle label for a given proposal id in the bundle directory.
##########################################################################################

import os
import pdslogger

from hst_helper.fs_utils import get_deliverable_path
from hst_helper.general_utils import (create_collection_label,
                                      get_mod_history_from_label)

BUNDLE_LABEL = 'bundle.xml'
BUNDLE_LABEL_TEMPLATE = 'BUNDLE_LABEL.xml'

def label_hst_bundle(proposal_id, data_dict, logger=None, testing=False):
    """With a given proposal id, create the bundle label. Return the path of the bundle
    label.

    Inputs:
        proposal_id    a proposal id.
        data_dict      a data dictionary used to create the label.
        logger         pdslogger to use; None for default EasyLogger.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.

    Returns:    the path of the newly created bundle label.
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
    deliverable_path = get_deliverable_path(proposal_id, testing)
    bundle_label_path = f'{deliverable_path}/{BUNDLE_LABEL}'
    mod_history = get_mod_history_from_label(bundle_label_path, version_id)

    # TODO: determine the version of each entry
    bundle_entries = []
    for col_name in os.listdir(deliverable_path):
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
    bundle_data_dict.update(data_dict)

    # Create bundle collection label
    return create_collection_label(proposal_id, 'bundle', bundle_data_dict,
                                   BUNDLE_LABEL, BUNDLE_LABEL_TEMPLATE,
                                   logger, testing)
