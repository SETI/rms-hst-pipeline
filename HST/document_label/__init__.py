##########################################################################################
# document_label/__init__.py
##########################################################################################
import datetime
import os
import pdslogger
import shutil

from hst_helper import (DOCUMENT_EXT,
                        PROGRAM_INFO_FILE)

from hst_helper.fs_utils import (get_program_dir_path,
                                 get_instrument_id,
                                 get_file_suffix)

from citations import Citation_Information
from xmltemplate import XmlTemplate

def label_hst_document_directory(proposal_id, logger):
    """With a given proposal id, create document directory in the final bundle.
    1. Create document directory.
    2. Move/copy proposal files over from pipeline directory.
    3. Create document xml label file

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

    formatted_proposal_id = str(proposal_id).zfill(5)

    # Create documents directory and move proposal files over
    logger.info(f'Create documents directory and move proposal files.')
    pipeline_dir = get_program_dir_path(proposal_id, None, root_dir='pipeline')
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    document_dir = bundles_dir + f'/document/{formatted_proposal_id}'

    # Create documents directory
    os.makedirs(document_dir, exist_ok=True)
    citation_info = None

    # Search for proposal files & program info file stored at pipeline directory
    # Move them to bundles directory and collect neccessary data for the label
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

    # Create label for proposal files using xmltemplate
    logger.info(f'Create label for proposal files using templates/DOCUMENT_LABEL.xml.')
    this_dir = os.path.dirname(os.path.abspath(__file__))
    template = this_dir + '/../templates/DOCUMENT_LABEL.xml'
    TEMPLATE = XmlTemplate(template)
    XmlTemplate.set_logger(logger)

    # Set xml label name
    label_path = document_dir + f'/{formatted_proposal_id}.xml'

    # Increase the minor version number if a proposal file exists in backups
    version_id = (1, 0)
    backups_dir = get_program_dir_path(proposal_id, None) + '/backups'
    try:
        for file in os.listdir(backups_dir):
            for name_info in proposal_files_li:
                if name_info[0] in file:
                    version_id = (version_id[0], version_id[1]+1)
    except FileNotFoundError:
        pass

    timetag = max(os.path.getmtime(__file__), os.path.getmtime(template))
    label_date = datetime.datetime.fromtimestamp(timetag).strftime("%Y-%m-%d")

    # Construct data dictionary used for the document label
    data_dict = {
        'prop_id': proposal_id,
        'collection_name': 'document',
        'citation_info': citation_info,
        'version_id': version_id,
        'label_date'     : label_date,
        'proposal_files_li': proposal_files_li,
    }

    logger.info(f'Insert data to the document xml label.')
    TEMPLATE.write(data_dict, label_path)
    if TEMPLATE.ERROR_COUNT == 1:
        logger.error('1 error encountered', label_path)
    elif TEMPLATE.ERROR_COUNT > 1:
        logger.error(f'{TEMPLATE.ERROR_COUNT} errors encountered', label_path)

    return
