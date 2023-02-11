##########################################################################################
# finalize_document/__init__.py
##########################################################################################
import csv
import datetime
import os
import pdslogger
import shutil

from hst_helper import (DOCUMENT_EXT,
                        PROGRAM_INFO_FILE)
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path)
from hst_helper.general_utils import (create_xml_label,
                                      create_csv,
                                      get_citation_info,
                                      get_instrument_id_set,
                                      get_mod_history_from_old_label)

from citations import Citation_Information
from product_labels.xml_support import get_modification_history

DOC_LABEL_TEMPLATE = 'DOCUMENT_LABEL.xml'
COL_DOC_LABEL_TEMPLATE = 'DOCUMENT_COLLECTION_LABEL.xml'
CSV_FILENAME = 'collection.csv'
COL_DOC_LABEL = 'collection.xml'

def label_hst_document_directory(proposal_id, logger):
    """With a given proposal id, create document directory in the final bundle.
    1. Create document directory.
    2. Move/copy proposal files over from pipeline directory.
    3. Create document label.
    4. Create document collection csv.
    5. Create document collection label.

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst document directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Create documents directory and move proposal files over
    logger.info(f'Create documents directory for proposal id: {proposal_id}.')
    pipeline_dir = get_program_dir_path(proposal_id, None, root_dir='pipeline')
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    document_dir = bundles_dir + f'/document/{formatted_proposal_id}'
    os.makedirs(document_dir, exist_ok=True)

    # Search for proposal files & program info file stored at pipeline directory
    # Move them to bundles directory and collect neccessary data for the label
    logger.info('Move over proposal files.')
    proposal_files_li = []
    for file in os.listdir(pipeline_dir):
        basename, _, ext = file.rpartition('.')
        if ext in DOCUMENT_EXT or file == PROGRAM_INFO_FILE:
            proposal_files_li.append((basename, file))
            fp = pipeline_dir + f'/{file}'

            # Move the proposal files and program info file to the documents directory
            shutil.copy(fp, document_dir + f'/{file}')
            # shutil.move(fp, document_dir + f'/{file}')

    # Collect data to construct data dictionary used for the document label
    # Get version id
    # Increase the minor version number if a proposal file exists in backups. Note: a
    # propsal file is in backups if it's different than the latest file via web query.
    version_id = (1, 0)
    backups_dir = get_program_dir_path(proposal_id, None) + '/backups'
    try:
        for file in os.listdir(backups_dir):
            for name_info in proposal_files_li:
                if name_info[0] in file:
                    version_id = (version_id[0], version_id[1]+1)
    except FileNotFoundError:
        pass

    # Get label date
    timetag = os.path.getmtime(__file__)
    label_date = datetime.datetime.fromtimestamp(timetag).strftime("%Y-%m-%d")

    # get citation info
    citation_info = get_citation_info(proposal_id, logger)
    # Get imstrument id
    inst_ids = get_instrument_id_set(proposal_id, logger)

    # Number of document inventory:
    # 2 handbooks per instrument + the proposal file directory
    records_num = len(inst_ids) * 2 + 1

    # Get the mod history for document collection label if it's already existed.
    col_doc_label_path = bundles_dir + f'/document/{COL_DOC_LABEL}'
    mod_history = get_mod_history_from_old_label(col_doc_label_path, version_id)

    data_dict = {
        'prop_id': proposal_id,
        'collection_name': 'document',
        'citation_info': citation_info,
        'version_id': version_id,
        'label_date': label_date,
        'proposal_files_li': proposal_files_li,
        'inst_id_li': list(inst_ids),
        'csv_filename': CSV_FILENAME,
        'records_num': records_num,
        'mod_history': mod_history,
    }

    # Create document label
    logger.info(f'Create label for proposal files using {DOC_LABEL_TEMPLATE}.')
    # Document label template path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    doc_template = this_dir + f'/../templates/{DOC_LABEL_TEMPLATE}'
    # Document label path
    doc_label = document_dir + f'/{formatted_proposal_id}.xml'
    create_xml_label(doc_template, doc_label, data_dict, logger)

    # Create document collection csv
    create_document_collection_csv(proposal_id, data_dict, logger)
    # Create document collection label
    create_document_collection_label(proposal_id, data_dict, logger)

def create_document_collection_csv(proposal_id, data_dict, logger):
    """With a given proposal id, create document collection csv in the final bundle.

    Inputs:
        proposal_id:    a proposal id.
        data_dict:      data dictonary to fill in the label template.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create document collection csv with proposal id: {proposal_id}')
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Set collection csv filename
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    document_collection_dir = bundles_dir + f'/document/{CSV_FILENAME}'

    # Construct collection data, each item in the list is a row in the csv file
    version_id = data_dict['version_id']
    document_lidvid = (f'P,urn:nasa:pds:hst_{formatted_proposal_id}'
                      + f':document:{formatted_proposal_id}'
                      + f'::{version_id[0]}.{version_id[1]}').split(',')
    collection_data = [document_lidvid]
    inst_ids = get_instrument_id_set(proposal_id, logger)
    for inst in inst_ids:
        inst = inst.lower()
        data_hb_lid = f'S,urn:nasa:pds:hst-support:document:{inst}-dhb\r\n'.split(',')
        collection_data.append(data_hb_lid)
        inst_hb_lid = f'S,urn:nasa:pds:hst-support:document:{inst}-ihb\r\n'.split(',')
        collection_data.append(inst_hb_lid)

    create_csv(document_collection_dir, collection_data, logger)

def create_document_collection_label(proposal_id, data_dict, logger):
    """With a given proposal id, create document collection label in the final bundle.

    Inputs:
        proposal_id:    a proposal id.
        data_dict:      data dictonary to fill in the label template.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create document collection csv with proposal id: {proposal_id}')

    # Create document collection label
    logger.info('Create label for document collection using '
                + f'templates/{COL_DOC_LABEL_TEMPLATE}.')
    # Document collection label template path
    col_doc_dir = os.path.dirname(os.path.abspath(__file__))
    col_doc_template = (col_doc_dir + f'/../templates/{COL_DOC_LABEL_TEMPLATE}')
    # Document collection label path
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    col_doc_label_path = bundles_dir + f'/document/{COL_DOC_LABEL}'

    create_xml_label(col_doc_template,col_doc_label_path, data_dict, logger)
