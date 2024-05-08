##########################################################################################
# finalize_context.py
#
# Create context directory, collection csv & xml, and investigation xml.
##########################################################################################

import pdslogger

from hst_helper.fs_utils import (create_col_dir_in_bundle,
                                 get_formatted_proposal_id)
from hst_helper.general_utils import (create_collection_label,
                                      create_csv,
                                      get_mod_history_from_label)

CSV_FILENAME = 'collection_context.csv'
COL_CTXT_LABEL = 'collection_context.xml'
COL_CTXT_LABEL_TEMPLATE = 'CONTEXT_COLLECTION_LABEL.xml'
INV_LABEL_TEMPLATE = 'INVESTIGATION_LABEL.xml'

def label_hst_context_directory(proposal_id, data_dict, logger=None, testing=False):
    """With a given proposal id, create context directory in the final bundle. These are
    the actions performed:

    1. Create context directory.
    2. Create context csv.
    3. create context xml label.
    4. create individual hst xml label.

    Inputs:
        proposal_id    a proposal id.
        data_dict      a data dictionary used to create the label.
        logger         pdslogger to use; None for default EasyLogger.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.

    Returns:    a tuple of the path of context collection label and the path of
                investigation label.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst context directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Create context directory
    logger.info(f'Create context directory for proposal id: {proposal_id}')
    _, context_dir = create_col_dir_in_bundle(proposal_id, 'context', testing)

    version_id = (1, 0)
    col_ctxt_label_path = f'{context_dir}/{COL_CTXT_LABEL}'
    mod_history = get_mod_history_from_label(col_ctxt_label_path, version_id)

    # Number of document inventory:
    # num of target ids + 3 (inst, inst_host, investigation)
    records_num = len(data_dict['target_identifications']) + 3

    ctx_data_dict = {
        'collection_name': 'context',
        'version_id': version_id,
        'csv_filename': CSV_FILENAME,
        'records_num': records_num,
        'mod_history': mod_history,
    }
    # ctx_data_dict.update(data_dict)
    ctx_data_dict = {**data_dict, **ctx_data_dict}


    # Create context collection csv
    create_context_collection_csv(proposal_id, context_dir, ctx_data_dict, logger)
    # Create context collection label
    ctxt_col_lbl = create_collection_label(proposal_id, 'context', ctx_data_dict,
                                           COL_CTXT_LABEL, COL_CTXT_LABEL_TEMPLATE,
                                           logger, testing)

    # Create investigation label
    inv_label = f'individual.hst_{formatted_proposal_id}.xml'
    inv_lbl = create_collection_label(proposal_id, 'context', ctx_data_dict,
                                      inv_label, INV_LABEL_TEMPLATE,
                                      logger, testing)

    return (ctxt_col_lbl, inv_lbl)

def create_context_collection_csv(proposal_id, context_dir, data_dict, logger=None):
    """With a given proposal id, path to context dir and data dictionary, create
    context collection csv in the final bundle.

    Inputs:
        proposal_id    a proposal id.
        context_dir    context directory path.
        data_dict      data dictonary to fill in the label template.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create context collection csv with proposal id: {proposal_id}')
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Set collection csv filename
    collection_context_csv = f'{context_dir}/{CSV_FILENAME}'

     # TODO: need to determine the version id for each entry in the csv
    collection_context_data = []
    for inst in data_dict['inst_id_li']:
        inst = inst.lower()
        inst_lidvid = f'S,urn:nasa:pds:context:instrument:hst.{inst}::1.0'.split(',')
        collection_context_data.append(inst_lidvid)

    # get target identification
    target_ids = data_dict['target_identifications']

    csv_entries = [
        'S,urn:nasa:pds:context:instrument_host:spacecraft.hst::1.0'.split(','),
        ('S,urn:nasa:pds:context:investigation:individual.'
         f'hst_{formatted_proposal_id}::1.0').split(',')
    ]

    for targ in target_ids:
        target = f'{targ["type"].lower()}.{targ["name"].lower()}'
        targ_ctxt_livid = f'S,urn:nasa:pds:context:target:{target}::1.0'.split(',')
        csv_entries.append(targ_ctxt_livid)
    collection_context_data += csv_entries

    create_csv(collection_context_csv, collection_context_data, logger)
