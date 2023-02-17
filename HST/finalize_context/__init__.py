##########################################################################################
# finalize_context/__init__.py
##########################################################################################
import csv
import datetime
import os
import pdslogger
import shutil

from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path)
from hst_helper.general_utils import (create_xml_label,
                                      create_csv,
                                      get_citation_info,
                                      get_instrument_id_set,
                                      get_mod_history_from_label,
                                      get_target_id_form_label)

CSV_FILENAME = 'collection_context.csv'
COL_CTXT_LABEL = 'collection_context.xml'
COL_CTXT_LABEL_TEMPLATE = 'CONTEXT_COLLECTION_LABEL.xml'

def label_hst_context_directory(proposal_id, logger):
    """With a given proposal id, create context directory in the final bundle.
    1. Create context directory.
    2. Create context csv.
    3. create context xml label.
    4. create individual hst xml label.


    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
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
    logger.info(f'Create context directory for proposal id: {proposal_id}.')
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    context_dir = bundles_dir + '/context'
    os.makedirs(context_dir, exist_ok=True)

    #  Collect data to construct data dictionary used for the context label
    citation_info = get_citation_info(proposal_id, logger)
    inst_ids = get_instrument_id_set(proposal_id, logger)

    # get target identification
    col_ctxt_label_path = context_dir + f'{COL_CTXT_LABEL}'
    target_info = get_target_id_form_label(proposal_id, col_ctxt_label_path)

     # Get label date
    timetag = os.path.getmtime(__file__)
    label_date = datetime.datetime.fromtimestamp(timetag).strftime("%Y-%m-%d")

    version_id = (1, 0)
    col_ctxt_label_path = context_dir + f'/{COL_CTXT_LABEL}'
    mod_history = get_mod_history_from_label(col_ctxt_label_path, version_id)

    # Number of document inventory:
    # num of target ids + 3 (inst, inst_host, investigation)
    records_num = len(target_info) + 3

    data_dict = {
        'prop_id': proposal_id,
        'collection_name': 'context',
        'citation_info': citation_info,
        'target_identifications': target_info,
        'version_id': version_id,
        'label_date': label_date,
        'inst_id_li': list(inst_ids),
        'csv_filename': CSV_FILENAME,
        'records_num': records_num,
        'mod_history': mod_history,
    }

    # Create context collection csv
    create_context_collection_csv(proposal_id, context_dir, data_dict, logger)
    # Create context collection label
    create_context_collection_label(proposal_id, data_dict, logger)


def create_context_collection_csv(proposal_id, context_dir, data_dict, logger):
    """With a given proposal id, path to context dir and data dictionary, create
    context collection csv in the final bundle.

    Inputs:
        proposal_id:    a proposal id.
        context_dir:    context directory path.
        data_dict:      data dictonary to fill in the label template.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create context collection csv with proposal id: {proposal_id}')
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Set collection csv filename
    collection_context_csv = context_dir + f'/{CSV_FILENAME}'

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
         + f'hst_{formatted_proposal_id}::1.0').split(',')
    ]

    for targ in target_ids:
        target = f'{targ["type"].lower()}.{targ["name"].lower()}'
        targ_ctxt_livid = f'S,urn:nasa:pds:context:target:{target}::1.0'.split(',')
        csv_entries.append(targ_ctxt_livid)
    collection_context_data += csv_entries

    create_csv(collection_context_csv, collection_context_data, logger)

def create_context_collection_label(proposal_id, data_dict, logger):
    """With a given proposal id, create context collection label in the final bundle.

    Inputs:
        proposal_id:    a proposal id.
        data_dict:      data dictonary to fill in the label template.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create context collection csv with proposal id: {proposal_id}')

    # Create context collection label
    logger.info('Create label for context collection using '
                + f'templates/{COL_CTXT_LABEL_TEMPLATE}.')
    # context collection label template path
    col_ctxt_dir = os.path.dirname(os.path.abspath(__file__))
    col_ctxt_template = (col_ctxt_dir + f'/../templates/{COL_CTXT_LABEL_TEMPLATE}')
    # context collection label path
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    col_ctxt_label_path = bundles_dir + f'/context/{COL_CTXT_LABEL}'

    create_xml_label(col_ctxt_template, col_ctxt_label_path, data_dict, logger)
