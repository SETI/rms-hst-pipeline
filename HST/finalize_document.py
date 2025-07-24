##########################################################################################
# finalize_document.py
#
# - Create document directory.
# - Move/copy proposal files over from pipeline directory.
# - Create document xml.
# - Create document collection csv & xml.
##########################################################################################

import os
import pdslogger
import shutil

from hst_helper import (DOCUMENT_EXT,
                        PROGRAM_INFO_FILE)
from hst_helper.fs_utils import (create_col_dir_in_bundle,
                                 get_deliverable_path,
                                 get_formatted_proposal_id,
                                 get_program_dir_path)
from hst_helper.general_utils import (create_xml_label,
                                      create_collection_label,
                                      create_csv,
                                      get_instrument_id_set,
                                      get_mod_history_from_label)

DOC_LABEL_TEMPLATE = 'DOCUMENT_LABEL.xml'
COL_DOC_LABEL_TEMPLATE = 'DOCUMENT_COLLECTION_LABEL.xml'
CSV_FILENAME = 'collection.csv'
COL_DOC_LABEL = 'collection.xml'

def finalize_hst_document_directory(proposal_id, data_dict, logger=None, testing=False):
    """With a given proposal id, create document directory in the final bundle. These
    are the actions performed:

    1. Create document directory.
    2. Move/copy proposal files over from pipeline directory.
    3. Create document label.
    4. Create document collection csv.
    5. Create document collection label.

    Inputs:
        proposal_id    a proposal id.
        data_dict      a data dictionary used to create the label.
        logger         pdslogger to use; None for default EasyLogger.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.

    Returns:    a tupel of the path of document collection label and the path of
                document label.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst document directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Create document directory and move proposal files over
    logger.info(f'Create document directory for proposal id: {proposal_id}')
    _, document_dir = create_col_dir_in_bundle(proposal_id, 'document', testing)

    # For testing purpose, data_dict is pre-constructed, no need to walk through
    # directories that might not exist.
    version_id = (1, 0)
    proposal_files_li = []
    if not testing: # pragma: no cover
        # Search for proposal files & program info file stored at pipeline directory
        # Move them to bundles directory and collect neccessary data for the label
        pipeline_dir = get_program_dir_path(proposal_id, None, root_dir='pipeline')
        logger.info('Move over proposal files')

        for file in os.listdir(pipeline_dir):
            basename, _, ext = file.rpartition('.')
            if ext in DOCUMENT_EXT or file == PROGRAM_INFO_FILE:
                proposal_files_li.append((basename, file))
                file_path = f'{pipeline_dir}/{file}'

                # Move the proposal files and program info file to the document directory
                shutil.copy(file_path, f'{document_dir}/{file}')
                # shutil.move(file_path, f'{document_dir}/{file}')

        # Collect data to construct data dictionary used for the document label
        # Get version id
        # Increase the minor version number if a proposal file exists in backups. Note: a
        # propsal file is in backups if it's different than the latest file via web query.
        backups_dir = get_program_dir_path(proposal_id, None) + '/backups'
        try:
            for file in os.listdir(backups_dir):
                for name_info in proposal_files_li:
                    if name_info[0] in file:
                        version_id = (version_id[0], version_id[1]+1)
        except FileNotFoundError:
            pass

    # Number of document inventory:
    # 2 handbooks per instrument + the proposal file directory
    records_num = len(data_dict['inst_id_li']) * 2 + 1

    # Get the mod history for document collection label if it's already existed.
    col_doc_label_path = f'{document_dir}/{COL_DOC_LABEL}'
    mod_history = get_mod_history_from_label(col_doc_label_path, version_id)

    doc_data_dict = {
        'collection_name': 'document',
        'version_id': version_id,
        'proposal_files_li': proposal_files_li,
        'csv_filename': CSV_FILENAME,
        'records_num': records_num,
        'mod_history': mod_history,
    }
    # doc_data_dict.update(data_dict)
    doc_data_dict = {**data_dict, **doc_data_dict}

    # Create document label
    logger.info(f'Create label for proposal files using {DOC_LABEL_TEMPLATE}')
    # Document label template path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    doc_template = f'{this_dir}/templates/{DOC_LABEL_TEMPLATE}'
    # Document label path
    doc_lbl = f'{document_dir}/{formatted_proposal_id}.xml'
    create_xml_label(doc_template, doc_lbl, doc_data_dict, logger)

    # Create document collection csv
    create_document_collection_csv(proposal_id, doc_data_dict, logger, testing)
    # Create document collection label
    doc_col_lbl = create_collection_label(proposal_id, 'document', doc_data_dict,
                                          COL_DOC_LABEL, COL_DOC_LABEL_TEMPLATE,
                                          logger, testing)

    return (doc_col_lbl, doc_lbl)

def create_document_collection_csv(proposal_id, data_dict, logger=None, testing=False):
    """With a given proposal id, create document collection csv in the final bundle.

    Inputs:
        proposal_id    a proposal id.
        data_dict      data dictonary to fill in the label template.
        logger         pdslogger to use; None for default EasyLogger.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create document collection csv with proposal id: {proposal_id}')
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Set collection csv filename
    deliverable_path = get_deliverable_path(proposal_id, testing)
    document_collection_dir = f'{deliverable_path}/document/{CSV_FILENAME}'

    # Construct collection data, each item in the list is a row in the csv file
    version_id = data_dict['version_id']
    document_lidvid = (f'P,urn:nasa:pds:hst_{formatted_proposal_id}'
                       f':document:{formatted_proposal_id}'
                       f'::{version_id[0]}.{version_id[1]}').split(',')
    collection_data = [document_lidvid]
    inst_ids = get_instrument_id_set(proposal_id, logger)
    for inst in inst_ids:
        inst = inst.lower()
        data_hb_lid = f'S,urn:nasa:pds:hst-support:document:{inst}-dhb'.split(',')
        collection_data.append(data_hb_lid)
        inst_hb_lid = f'S,urn:nasa:pds:hst-support:document:{inst}-ihb'.split(',')
        collection_data.append(inst_hb_lid)

    create_csv(document_collection_dir, collection_data, logger)
