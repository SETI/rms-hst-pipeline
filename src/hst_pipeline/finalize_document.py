##########################################################################################
# finalize_document.py
#
# - Create document directory.
# - Copy the proposal files (or, for early programs without them, the MAST proposal
#   search page) over from the pipeline directory.
# - Copy the handbooks whose labels name one of the program's instruments as a keyword,
#   with their labels, from the HST handbooks bundle ($HST_HANDBOOKS).
# - Create document xml.
# - Create document collection csv & xml; the handbooks are secondary members.
##########################################################################################

import datetime
import os
import pdslogger
import re
import shutil
from xml.etree import ElementTree

from hst_pipeline.hst_helper import (DOCUMENT_EXT,
                                     DOCUMENT_EXT_FOR_CITATION_INFO,
                                     DOCUMENT_STANDARD_IDS,
                                     HANDBOOK_KINDS,
                                     HANDBOOK_LID,
                                     HANDBOOKS_COLLECTION,
                                     HANDBOOKS_ENV,
                                     LABEL_SUFFIX,
                                     MAST_PROPOSAL_SUFFIX,
                                     MAST_PROPOSAL_URL)
from hst_pipeline.hst_helper.fs_utils import (BACKUP_TIME_PATTERN,
                                              create_col_dir_in_bundle,
                                              get_deliverable_path,
                                              get_formatted_proposal_id,
                                              get_program_dir_path)
from hst_pipeline.hst_helper.general_utils import (create_xml_label,
                                                   create_collection_label,
                                                   create_csv,
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

class HandbooksError(RuntimeError):
    """Raised when the HST handbooks bundle cannot be found or read."""


def finalize_hst_document_directory(proposal_id, data_dict, logger=None, testing=False,
                                    handbooks_dir=None):
    """With a given proposal id, create document directory in the final bundle. These
    are the actions performed:

    1. Create document directory.
    2. Copy proposal files (or the MAST proposal page) from pipeline directory.
    3. Copy the handbooks whose labels name one of the program's instruments as a
       keyword, and their labels, from the handbooks collection.
    4. Create document label.
    5. Create document collection csv, with the handbooks as secondary members.
    6. Create document collection label.

    Inputs:
        proposal_id      a proposal id.
        data_dict        a data dictionary used to create the label; its 'inst_id_li'
                         lists the program's instruments.
        logger           pdslogger to use; None for default EasyLogger.
        testing          the flag used to determine if we are calling the function for
                         testing purpose with the test directory.
        handbooks_dir    the handbooks collection directory; None for the one in the
                         bundle named by $HST_HANDBOOKS (see get_handbooks_dir()).

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
        # The minor version counts the downloads that replaced a proposal file; see
        # count_document_revisions.
        backups_dir = get_program_dir_path(proposal_id, None) + '/backups'
        version_id = (1, count_document_revisions(backups_dir, proposal_files_li))

    # Copy the handbooks of the program's instruments, which are secondary members
    handbooks_dir = handbooks_dir or get_handbooks_dir()
    (handbook_lids, handbook_files) = get_handbook_files(data_dict['inst_id_li'],
                                                         handbooks_dir)
    logger.info(f'Copy {len(handbook_files)} handbook files from {handbooks_dir}')
    for lid in sorted(set(expected_handbook_lids(data_dict['inst_id_li']))
                      - set(handbook_lids)):
        logger.warn(f'Handbook {lid} is referenced but not in {handbooks_dir}')
    copy_handbook_files(handbook_files, handbooks_dir, document_dir, logger)

    # Number of document inventory: the proposal files + the handbooks
    records_num = len(handbook_lids) + 1

    # Get the mod history for document collection label if it's already existed.
    col_doc_label_path = f'{document_dir}/{COL_DOC_LABEL}'
    mod_history = get_mod_history_from_label(col_doc_label_path, version_id)

    doc_data_dict = {
        'collection_name': 'document',
        'version_id': version_id,
        'proposal_files_li': proposal_files_li,
        'source_li': source_li,
        'handbook_lids': handbook_lids,
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

def count_document_revisions(backups_dir, proposal_files_li):
    """Return the number of downloads that replaced any of the proposal files.

    hst-get-program-info moves each replaced file to backups/ as
    "<basename>-<ymdhms>.<ext>", with one time for all the files of one download, so
    each distinct time is one revision of the document.

    Input:
        backups_dir         the program's backups directory, which may not exist.
        proposal_files_li   the tuples (basename, file name, standard id) of the proposal
                            files, as returned by get_document_files().

    Returns:    the number of distinct backup times among those files' backups.
    """
    if not os.path.isdir(backups_dir):
        return 0
    patterns = []
    for (_, file, _) in proposal_files_li:
        (basename, _, ext) = file.partition('.')
        patterns.append(re.compile(re.escape(basename) + f'-({BACKUP_TIME_PATTERN})\\.'
                                   + re.escape(ext) + '$'))
    times = set()
    for name in os.listdir(backups_dir):
        for pattern in patterns:
            match = pattern.match(name)
            if match:
                times.add(match.group(1))
    return len(times)

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

def get_handbooks_dir():
    """Return the handbooks collection directory: the document/ directory of the HST
    handbooks bundle named by the HST_HANDBOOKS environment variable, or that directory
    itself if it has no document/ subdirectory.

    Returns:    the path to the directory holding the handbooks and their labels.

    Raises:
        HandbooksError    if HST_HANDBOOKS is unset or names no directory.
    """
    bundle_dir = os.environ.get(HANDBOOKS_ENV, '').strip()
    if not bundle_dir:
        raise HandbooksError(f'{HANDBOOKS_ENV} must name the HST handbooks bundle, '
                             'whose handbooks go in every document collection')

    collection_dir = os.path.join(bundle_dir, HANDBOOKS_COLLECTION)
    if os.path.isdir(collection_dir):
        return collection_dir
    if os.path.isdir(bundle_dir):
        return bundle_dir
    raise HandbooksError(f'{HANDBOOKS_ENV} directory not found: {bundle_dir}')

def get_handbook_instrument(filename):
    """Return the part of a handbook file's name before the first dash, in upper case;
    e.g., 'WFPC2' for WFPC2-Data-Handbook-2010-v5-final.pdf and 'HST' for
    hst-data-handbook-volume-1.lblx. Every handbook file's name has a dash; the proposal
    files' and collection files' names don't.

    Input:
        filename    a file name in the handbooks or a document collection.

    Returns:    the part before the dash, or None if the name has no dash.
    """
    (name, dash, _) = filename.partition('-')
    return name.upper() if dash else None

def expected_handbook_lids(inst_ids):
    """Return the LIDs of the handbooks that the bundle, collection, and product
    labels refer to for a set of instruments: a data handbook and an instrument handbook
    for each.

    Input:
        inst_ids    an iterable of instrument ids, e.g., ['NICMOS', 'WFPC'].

    Returns:    a list of LIDs.
    """
    return [HANDBOOK_LID.format(inst=inst.lower(), kind=kind)
            for inst in inst_ids for kind in HANDBOOK_KINDS]

def read_handbook_label(path):
    """Read a label in the handbooks collection.

    Input:
        path    path to the label.

    Returns:    None if the label is not a Product_Document (e.g., the collection
                label); otherwise a tuple (lid, keywords, files): the LID, the set of
                Citation_Information keywords in upper case, and the names of the
                document files the label describes.

    Raises:
        HandbooksError    if the label cannot be parsed or has no logical_identifier.
    """
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as e:
        raise HandbooksError(f'Cannot parse {path}: {e}')

    def local(tag):
        return tag.rpartition('}')[2]

    if local(root.tag) != 'Product_Document':
        return None

    lid = None
    keywords = set()
    files = []
    for elem in root.iter():
        tag = local(elem.tag)
        text = (elem.text or '').strip()
        if tag == 'logical_identifier' and lid is None:
            lid = text
        elif tag == 'Citation_Information':
            keywords |= {(k.text or '').strip().upper() for k in elem
                         if local(k.tag) == 'keyword'}
        elif tag == 'file_name' and text:
            files.append(text)

    if not lid:
        raise HandbooksError(f'No logical_identifier in {path}')
    return (lid, keywords, files)

def get_handbook_files(inst_ids, handbooks_dir):
    """Return the handbook files for a set of instruments, and the LIDs of their
    labels. A handbook belongs to an instrument if one of the keywords in its label's
    Citation_Information is that instrument's id, ignoring case; a handbook covering
    several instruments, such as a volume of the HST Data Handbook, names each of them.
    Its files are the label and the documents the label names.

    Input:
        inst_ids         an iterable of instrument ids, e.g., ['NICMOS', 'WFPC'].
        handbooks_dir    the handbooks collection directory.

    Returns:    a tuple (lids, files): the sorted LIDs of the handbook labels, and the
                sorted names of all the handbook files (documents and labels).

    Raises:
        HandbooksError    if a handbook label cannot be read, has no
                          logical_identifier, or names a document that is missing.
    """
    names = {inst.upper() for inst in inst_ids}

    lids = []
    files = []
    for label in sorted(os.listdir(handbooks_dir)):
        path = os.path.join(handbooks_dir, label)
        if not label.endswith(LABEL_SUFFIX) or not os.path.isfile(path):
            continue
        info = read_handbook_label(path)
        if info is None:
            continue
        (lid, keywords, documents) = info
        if not keywords & names:
            continue
        for document in documents:
            if not os.path.isfile(os.path.join(handbooks_dir, document)):
                raise HandbooksError(f'{path} names a missing file: {document}')
        lids.append(lid)
        files += [label] + documents

    return (sorted(lids), sorted(set(files)))

def copy_handbook_files(files, handbooks_dir, document_dir, logger=None):
    """Copy handbook files into a document collection, skipping those already there
    with the same size and modification time. Handbook files left in the collection from
    an earlier run that are no longer wanted, because an instrument has been dropped or a
    handbook renamed, are removed: every file whose name has a dash, as handbook names
    do and the proposal files' and collection files' names don't.

    Input:
        files            names of the handbook files to copy.
        handbooks_dir    the handbooks collection directory.
        document_dir     the program's document collection directory.
        logger           pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    for file in files:
        src = os.path.join(handbooks_dir, file)
        dst = os.path.join(document_dir, file)
        if os.path.isfile(dst):
            (src_stat, dst_stat) = (os.stat(src), os.stat(dst))
            if (src_stat.st_size == dst_stat.st_size
                and int(src_stat.st_mtime) == int(dst_stat.st_mtime)):
                continue
        shutil.copy2(src, dst)

    wanted = set(files)
    for file in os.listdir(document_dir):
        if file not in wanted and get_handbook_instrument(file):
            logger.info(f'Remove handbook file no longer needed: {file}')
            os.remove(os.path.join(document_dir, file))

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
        data_dict      data dictonary to fill in the label template; its
                       'handbook_lids' lists the LIDs of the secondary members.
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

    # The handbooks are secondary members, referenced by LID
    for lid in data_dict['handbook_lids']:
        collection_data.append(['S', lid])

    create_csv(document_collection_dir, collection_data, logger)
