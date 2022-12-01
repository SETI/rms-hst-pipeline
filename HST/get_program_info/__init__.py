##########################################################################################
# get_program_info/__init__.py
##########################################################################################
import os
import pdslogger
import fs.path

import urllib.error
import urllib.parse
import urllib.request

from hst_helper.fs_utils import (backup_file,
                                 get_program_dir_path)

from hst_helper import (DOCUMENT_SUFFIXES,
                        DOCUMENT_SUFFIXES_FOR_CITATION_INFO)

from citations import Citation_Information

def get_program_info(proposal_id, download_dir=None, logger=None):
    """
    Download proposal files and generate program-info.txt for the given proposal ID
    Input:
        proposal_id:    a proposal id.
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
        proposal_id:    a proposal id.
        url:            the url to retrieve the text of a proposal file
        filepath:       the file path of the existing proposal file or the file path used
                        to store the newly retrieved proposal file.
    """
    logger = logger or pdslogger.EasyLogger()

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
    program_info_filepath = program_dir + '/program-info.txt'
    if not os.path.exists(program_info_filepath):
        create_program_info_file(filepath)

    return False

def is_proposal_file_different(new_contents, filepath):
    """
    Return a boolean flag to determine if a proposal file needs to be replaced/created.
    Input:
        contents:   the contents of the newly retrieved proposal file.
        filepath:   the file path of the existing proposal file or the file path used to
                    store the newly retrieved proposal file.
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            file_contents = f.read()
        if file_contents == new_contents:
            return False
        else:
            return True
    else:
        return True

def create_program_info_file(filepath):
    """Create and store citation info in program-info.txt
    Input:
        filepath:   the file path of a proposal file used to get the citation info.
    """
    citation_info = Citation_Information.create_from_file(filepath)
    program_dir, _, _ = filepath.rpartition('/')
    program_info_filepath = program_dir + '/program-info.txt'
    citation_info.write(program_info_filepath)

def download_proposal_files(proposal_id, download_dir, logger=None):
    """
    Download proposal files for the given proposal ID into a directory and return a
    set of the basenames of the files successfully downloaded.
    Input:
        proposal_id:    a proposal id.
        download_dir:   the directory to store proposal files
    """
    logger = logger or pdslogger.EasyLogger()
    # A table contains a list of tuple (url for a proposal file, stored file name)
    table = [
        (f'https://www.stsci.edu/hst/phase2-public/{proposal_id}.{suffix}',
         str(proposal_id).zfill(5)+f'.{suffix}') for suffix in DOCUMENT_SUFFIXES
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
                ext in DOCUMENT_SUFFIXES_FOR_CITATION_INFO):
                logger.info(f'Create program info file from {basename}')
                create_program_info_file(filepath)
                is_program_info_file_created = True
    logger.close()

    return res
