##########################################################################################
# finalize_document.py
#
# - Create document directory.
# - Copy the proposal files (or, for early programs without them, the MAST proposal
#   search page) over from the pipeline directory.
# - Create document xml.
# - Create document collection csv & xml.
##########################################################################################

import datetime
import os
import pdslogger
import re
import shutil

from hst_pipeline.hst_helper import (DOCUMENT_EXT,
                                     DOCUMENT_EXT_FOR_CITATION_INFO,
                                     DOCUMENT_STANDARD_IDS,
                                     LABEL_SUFFIX,
                                     MAST_PROPOSAL_SUFFIX,
                                     MAST_PROPOSAL_URL)
from hst_pipeline.hst_helper.fs_utils import (create_col_dir_in_bundle,
                                              get_deliverable_path,
                                              get_formatted_proposal_id,
                                              get_program_dir_path)
from hst_pipeline.hst_helper.general_utils import (create_xml_label,
                                                   create_collection_label,
                                                   create_csv,
                                                   get_instrument_id_set,
                                                   get_mod_history_from_label)

DOC_LABEL_TEMPLATE = 'DOCUMENT_LABEL' + LABEL_SUFFIX
COL_DOC_LABEL_TEMPLATE = 'DOCUMENT_COLLECTION_LABEL' + LABEL_SUFFIX
CSV_FILENAME = 'collection.csv'
COL_DOC_LABEL = 'collection' + LABEL_SUFFIX

# In a MAST proposal search page, the proposal content starts at the centered title and
# ends at the close of the content <div>, after the list of publications; before it are
# the site header and menus, after it the data-search links and the site footer.
MAST_PAGE_START_REGEX = re.compile(r"<h2 align='center'>", re.I)
MAST_PAGE_END_REGEX = re.compile(r'</div>', re.I)

def finalize_hst_document_directory(proposal_id, data_dict, logger=None, testing=False):
    """With a given proposal id, create document directory in the final bundle. These
    are the actions performed:

    1. Create document directory.
    2. Copy proposal files (or the MAST proposal page) from pipeline directory.
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
    source_li = []
    if not testing: # pragma: no cover
        # Search for proposal files stored at pipeline directory
        # Move them to bundles directory and collect neccessary data for the label
        pipeline_dir = get_program_dir_path(proposal_id, None, root_dir='pipeline')
        logger.info('Move over proposal files')

        proposal_files_li = get_document_files(proposal_id, pipeline_dir)
        for (_, file, _) in proposal_files_li:
            # Copy the proposal files to the document directory; the MAST page loses
            # the site's header and footer
            if file.endswith(MAST_PROPOSAL_SUFFIX):
                copy_mast_proposal_page(f'{pipeline_dir}/{file}',
                                        f'{document_dir}/{file}', logger)
            else:
                shutil.copy(f'{pipeline_dir}/{file}', f'{document_dir}/{file}')

        source_li = get_document_sources(proposal_id, pipeline_dir, proposal_files_li)

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
        'source_li': source_li,
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
    doc_lbl = f'{document_dir}/{formatted_proposal_id}{LABEL_SUFFIX}'
    create_xml_label(doc_template, doc_lbl, doc_data_dict, logger)

    # Create document collection csv
    create_document_collection_csv(proposal_id, doc_data_dict, logger, testing)
    # Create document collection label
    doc_col_lbl = create_collection_label(proposal_id, 'document', doc_data_dict,
                                          COL_DOC_LABEL, COL_DOC_LABEL_TEMPLATE,
                                          logger, testing)

    return (doc_col_lbl, doc_lbl)

def get_document_files(proposal_id, pipeline_dir):
    """Return the files in a program's pipeline directory that belong in the bundle's
    document collection: the proposal files (.apt, .pdf, .pro, .prop) and, for early
    programs that have none, the MAST proposal search page <nnnnn>.html.

    Input:
        proposal_id     a proposal id.
        pipeline_dir    the program's pipeline directory.

    Returns:    a list of tuples (basename without extension, file name, PDS4
                document_standard_id), sorted by file name.
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    mast_page = f'{formatted_proposal_id}{MAST_PROPOSAL_SUFFIX}'
    doc_files = []
    for file in sorted(os.listdir(pipeline_dir)):
        basename, _, ext = file.rpartition('.')
        if basename != formatted_proposal_id:
            continue
        if ext in DOCUMENT_EXT:
            doc_files.append((basename, file, DOCUMENT_STANDARD_IDS[ext]))

    # The MAST page is only the fallback, used when there is no .apt or .pro file
    has_citation_file = any(file.rpartition('.')[2] in DOCUMENT_EXT_FOR_CITATION_INFO
                            for (_, file, _) in doc_files)
    if not has_citation_file and os.path.isfile(f'{pipeline_dir}/{mast_page}'):
        basename, _, ext = mast_page.rpartition('.')
        doc_files.append((basename, mast_page, DOCUMENT_STANDARD_IDS[ext]))

    return doc_files

def get_document_sources(proposal_id, pipeline_dir, doc_files):
    """Return the source of each document file that is not a proposal file, i.e., the
    MAST proposal search page, for an External_Reference in the document label.

    Input:
        proposal_id     a proposal id.
        pipeline_dir    the program's pipeline directory.
        doc_files       the list returned by get_document_files().

    Returns:    a list of tuples (file name, URL, retrieval date as yyyy-mm-dd); the date
                is the modification date of the page saved in the pipeline directory.
    """
    sources = []
    for (_, file, _) in doc_files:
        if not file.endswith(MAST_PROPOSAL_SUFFIX):
            continue
        url = MAST_PROPOSAL_URL.format(int(proposal_id))
        mtime = os.path.getmtime(f'{pipeline_dir}/{file}')
        date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
        sources.append((file, url, date))

    return sources

def strip_mast_proposal_page(page):
    """Return the proposal content of a MAST proposal search page: a horizontal rule,
    then everything from the title through the list of publications.

    Input:
        page    the text of the page.

    Returns:    the stripped text, or None if the title cannot be found.
    """
    start = MAST_PAGE_START_REGEX.search(page)
    if not start:
        return None
    end = MAST_PAGE_END_REGEX.search(page, start.end())
    content = page[start.start():end.start() if end else len(page)]
    return '<hr>\n' + content.rstrip() + '\n'

def copy_mast_proposal_page(src, dst, logger=None):
    """Copy a MAST proposal search page into the document collection, without the
    site's header and footer. If the proposal content cannot be found, copy the page
    unchanged.

    Input:
        src       path to the page saved in the pipeline directory.
        dst       path of the copy in the document collection.
        logger    pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    # The pages are ISO-8859-1; Latin-1 also passes any other byte through unchanged
    with open(src, encoding='latin-1') as f:
        page = f.read()

    stripped = strip_mast_proposal_page(page)
    if stripped is None:
        logger.warn(f'No proposal title found in {src}; copied unchanged')
        shutil.copy(src, dst)
        return

    with open(dst, 'w', encoding='latin-1') as f:
        f.write(stripped)

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
