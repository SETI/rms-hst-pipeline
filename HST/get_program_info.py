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
##########################################################################################

import fs.path
import os
import pdslogger
import urllib.error
import urllib.parse
import urllib.request

from citations import Citation_Information
from hst_helper import (DOCUMENT_EXT,
                        DOCUMENT_EXT_FOR_CITATION_INFO,
                        PROGRAM_INFO_FILE)
from hst_helper.fs_utils import (backup_file,
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
        resp = urllib.request.urlopen(url)
        new_contents = resp.read()
    except urllib.error.URLError as e:
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
        filepath = fs.path.join(download_dir, basename)
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

    return res
