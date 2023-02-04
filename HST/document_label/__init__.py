##########################################################################################
# document_label/__init__.py
##########################################################################################
import csv
import datetime
import os
import pdslogger
import shutil

from hst_helper import (DOCUMENT_EXT,
                        PROGRAM_INFO_FILE)

from hst_helper import INST_ID_DICT
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path,
                                 get_instrument_id)
from hst_helper.general_utils import create_xml_label

from citations import Citation_Information
from product_labels.xml_support import get_modification_history

DOC_LABEL_TEMPLATE = 'DOCUMENT_LABEL.xml'
DOC_COL_LABEL_TEMPLATE = 'DOCUMENT_COLLECTION_LABEL.xml'
CSV_FILENAME = 'collection.csv'
DOC_COL_LABEL = 'collection.xml'

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

    # Create documents directory
    os.makedirs(document_dir, exist_ok=True)

    # Search for proposal files & program info file stored at pipeline directory
    # Move them to bundles directory and collect neccessary data for the label
    logger.info('Move over proposal files.')
    citation_info = None
    proposal_files_li = []
    for file in os.listdir(pipeline_dir):
        basename, _, ext = file.rpartition('.')
        if ext in DOCUMENT_EXT or file == PROGRAM_INFO_FILE:
            proposal_files_li.append((basename, file))
            fp = pipeline_dir + f'/{file}'
            if citation_info is None:
                citation_info = Citation_Information.create_from_file(fp)
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

    # Get imstrument id
    if formatted_proposal_id not in INST_ID_DICT:
        # Walk through all the downloaded files from MAST in staging directory
        files_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
        for root, dirs, files in os.walk(files_dir):
            for file in files:
                inst_id = get_instrument_id(file)
                formatted_proposal_id = str(proposal_id).zfill(5)
                if inst_id is not None:
                    INST_ID_DICT[formatted_proposal_id].add(inst_id)

    # Number of document inventory:
    # 2 handbooks per instrument + the proposal file directory
    records_num = len(INST_ID_DICT[formatted_proposal_id]) * 2 + 1

    # Get the mod history for document collection label if it's already existed.
    doc_col_label_path = bundles_dir + f'/document/{DOC_COL_LABEL}'
    mod_history = []
    if os.path.exists(doc_col_label_path):
        with open(doc_col_label_path) as f:
            xml_content = f.read()
            modification_history = get_modification_history(xml_content)
            old_version = modification_history[-1][1]
            if old_version != version_id:
                mod_history = modification_history

    data_dict = {
        'prop_id': proposal_id,
        'collection_name': 'document',
        'citation_info': citation_info,
        'version_id': version_id,
        'label_date'     : label_date,
        'proposal_files_li': proposal_files_li,
        'inst_id_li': list(INST_ID_DICT[formatted_proposal_id]),
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

     # Create collection csv file
    if formatted_proposal_id not in INST_ID_DICT:
        # Walk through all the downloaded files from MAST in staging directory
        files_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
        for root, dirs, files in os.walk(files_dir):
            for file in files:
                inst_id = get_instrument_id(file)
                formatted_proposal_id = str(proposal_id).zfill(5)
                if inst_id is not None:
                    INST_ID_DICT[formatted_proposal_id].add(inst_id)

    # Set collection csv filename
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    document_collection_dir = bundles_dir + f'/document/{CSV_FILENAME}'

    # Construct collection data, each item in the list is a row in the csv file
    version_id = data_dict['version_id']
    document_lidvid = (f'P,urn:nasa:pds:hst_{formatted_proposal_id}'
                      + f':document:{formatted_proposal_id}'
                      + f'::{version_id[0]}.{version_id[1]}').split(',')
    collection_data = [document_lidvid]
    for inst in INST_ID_DICT[formatted_proposal_id]:
        inst = inst.lower()
        data_hb_lid = f'S,urn:nasa:pds:hst-support:document:{inst}-dhb\r\n'.split(',')
        collection_data.append(data_hb_lid)
        inst_hb_lid = f'S,urn:nasa:pds:hst-support:document:{inst}-ihb\r\n'.split(',')
        collection_data.append(inst_hb_lid)

    # open the file in the write mode
    with open(document_collection_dir, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write rows to the csv file
        writer.writerows(collection_data)

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
                + f'templates/{DOC_COL_LABEL_TEMPLATE}.')
    # Document collection label template path
    doc_col_dir = os.path.dirname(os.path.abspath(__file__))
    doc_col_template = (doc_col_dir + f'/../templates/{DOC_COL_LABEL_TEMPLATE}')
    # Document collection label path
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    doc_col_label_path = bundles_dir + f'/document/{DOC_COL_LABEL}'

    create_xml_label(doc_col_template,doc_col_label_path, data_dict, logger)
