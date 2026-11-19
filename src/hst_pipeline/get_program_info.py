##########################################################################################
# get_program_info.py
#
# get_program_info is the main function called in get_program_info pipeline task script.
# It will do these actions:
#
# - Download the proposal files via a web query.
# - If these files are the same as the existing ones, return.
# - Otherwise:
#   - Rename each existing file by appending “-<ymdhms>” before its extension and moving
#     it to a backups/ subdirectory.
#   - Save the newly downloaded files.
# - Generate the new program-info.txt with a possibly modified citation.
# - If there is no .apt or .pro file (as for early programs), fall back to the MAST
#   proposal search page for the citation info.
##########################################################################################

import os
import pdslogger
from socket import timeout
import urllib.error
import urllib.parse
import urllib.request

from hst_pipeline.citations import Citation_Information
from hst_pipeline.hst_helper import (DOCUMENT_EXT,
                                     DOCUMENT_EXT_FOR_CITATION_INFO,
                                     MAST_PROPOSAL_SUFFIX,
                                     MAST_PROPOSAL_URL,
                                     PROGRAM_INFO_FILE)
from hst_pipeline.hst_helper.fs_utils import (backup_file,
                                              get_formatted_proposal_id,
                                              get_program_dir_path)

def get_program_info(proposal_id, download_dir=None, logger=None):
    """Download proposal files and generate PROGRAM_INFO_FILE for the given proposal ID.

    Input:
        proposal_id    a proposal id.

    Returns:    a set of the basenames of the files successfully downloaded.
    """
    logger = logger or pdslogger.EasyLogger()

    download_dir = download_dir or get_program_dir_path(proposal_id)
    # remove the leading zero's of proposal_id if there is one. The proposal file name
    # does not have the leading zero.
    proposal_id = int(proposal_id)
    logger.info(f'Download proposal files for {proposal_id}')
    res = download_proposal_files(proposal_id, download_dir, logger)
    return res

def is_proposal_file_retrieved(proposal_id, url, filepath, logger=None):
    """Return a boolean flag to determine if a proposal file is retrieved.

    Input:
        proposal_id    a proposal id.
        url            the url to retrieve the text of a proposal file
        filepath       the file path of the existing proposal file or the file path used
                       to store the newly retrieved proposal file.

    Returns:    a boolean that indicates if a proposal file is retrieved.
    """
    logger = logger or pdslogger.EasyLogger()

    # Check if a proposal file is retrieved
    try:
        resp = urllib.request.urlopen(url, timeout=600)
        new_contents = resp.read()
    except urllib.error.URLError as e:
        if isinstance(e.reason, timeout):
            logger.error(f'Timeout Error. \nURL: {url}')
        else:
            logger.info(f'file from {url} is not found for {proposal_id}')
        return False

    if is_proposal_file_different(new_contents, filepath):
        # Back up the current proposal file if there is a different one
        if os.path.exists(filepath):
            backup_file(proposal_id, None, filepath)
        # Save the new proposal file
        with open(filepath, 'wb') as f:
            f.write(new_contents)
        return True

    # For the case when all propsal files are downloaded but program info file
    # doesn't exist.
    program_dir, _, _ = filepath.rpartition('/')
    program_info_filepath = f'{program_dir}/{PROGRAM_INFO_FILE}'
    if not os.path.exists(program_info_filepath):
        create_program_info_file(filepath)

    return False

def is_proposal_file_different(new_contents, filepath):
    """Return a boolean flag to determine if a proposal file needs to be replaced/created.

    Input:
        contents    the contents of the newly retrieved proposal file.
        filepath    the file path of the existing proposal file or the file path used to
                    store the newly retrieved proposal file.

    Returns:    a boolean that indicates if a proposal file needs to be replaced/created.
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            file_contents = f.read()
        return file_contents != new_contents
    else:
        return True

def create_program_info_file(filepath):
    """Create and store citation info in PROGRAM_INFO_FILE.

    Input:
        filepath    the file path of a proposal file used to get the citation info.
    """
    citation_info = Citation_Information.create_from_file(filepath)
    program_dir, _, _ = filepath.rpartition('/')
    program_info_filepath = f'{program_dir}/{PROGRAM_INFO_FILE}'
    citation_info.write(program_info_filepath)

def download_proposal_files(proposal_id, download_dir, logger=None):
    """Download proposal files for the given proposal ID into a directory and return a
    set of the basenames of the files successfully downloaded.

    Input:
        proposal_id     a proposal id.
        download_dir    the directory to store proposal files.

    Returns:    a set of the basenames of the files successfully downloaded.
    """
    logger = logger or pdslogger.EasyLogger()
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    # A table contains a list of tuple (url for a proposal file, stored file name)
    table = [
        (f'https://www.stsci.edu/hst/phase2-public/{proposal_id}.{suffix}',
         f'{formatted_proposal_id}.{suffix}') for suffix in DOCUMENT_EXT
    ]

    res = set()
    logger.open('Download proposal files')
    is_program_info_file_created = False
    for (url, basename) in table:
        filepath = os.path.join(download_dir, basename)
        # Download the new proposal files if necessary
        if is_proposal_file_retrieved(proposal_id, url, filepath, logger=logger):
            logger.info(f'Retrieve {basename} from {url}')
            res.add(basename)
            # Create or update program info file
            _, _, ext = basename.rpartition('.')
            if (not is_program_info_file_created and
                ext in DOCUMENT_EXT_FOR_CITATION_INFO):
                logger.info(f'Create program info file from {basename}')
                create_program_info_file(filepath)
                is_program_info_file_created = True

    # Early programs have no .apt or .pro file; use the MAST proposal search page instead
    has_citation_file = (is_program_info_file_created or
                         any(os.path.exists(os.path.join(download_dir, basename))
                             for (_, basename) in table
                             if basename.rpartition('.')[2]
                             in DOCUMENT_EXT_FOR_CITATION_INFO))
    if not has_citation_file:
        basename = download_mast_proposal_page(proposal_id, download_dir, logger)
        if basename:
            res.add(basename)

    return res

def download_mast_proposal_page(proposal_id, download_dir, logger=None):
    """Download the MAST proposal search page of a program and create PROGRAM_INFO_FILE
    from it; the fallback for programs with no .apt or .pro file.

    Input:
        proposal_id     a proposal id.
        download_dir    the directory to store the page.
        logger          pdslogger to use; None for default EasyLogger.

    Returns:    the basename of the page if it was newly downloaded or changed and
                its citation info could be read; otherwise None.
    """
    logger = logger or pdslogger.EasyLogger()
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    url = MAST_PROPOSAL_URL.format(int(proposal_id))
    basename = f'{formatted_proposal_id}{MAST_PROPOSAL_SUFFIX}'
    filepath = os.path.join(download_dir, basename)

    logger.info(f'No .apt or .pro file for {proposal_id}; using {url}')
    try:
        retrieved = is_proposal_file_retrieved(proposal_id, url, filepath, logger=logger)
        if not retrieved:
            return None
        logger.info(f'Create program info file from {basename}')
        create_program_info_file(filepath)
    except ValueError as e:
        # The page has no proposal information, e.g., MAST has no such proposal
        logger.error(f'No citation info for {proposal_id} in {url}: {e}')
        if os.path.exists(filepath):
            os.remove(filepath)
        return None

    return basename
