##########################################################################################
# hst_helper/fs_utils.py
##########################################################################################
import datetime
import os
import shutil

from hashlib import md5

from . import HST_DIR
from product_labels.suffix_info import INSTRUMENT_FROM_LETTER_CODE

def create_program_dir(proposal_id, visit=None, root_dir='pipeline'):
    """Create the program directory for IPPPSSOOT from a proposal id, return the path of
    the directory. If visit is specified, create the visit directory as well.

    Input:
        proposal_id:    a proposal id.
        visit:          the two character designation for the HST visit
        root_dir:       root directory of the program, it's either 'staging', 'pipeline'
                        or 'bundles'.
    """
    program_dir = get_program_dir_path(proposal_id, visit, root_dir)
    os.makedirs(program_dir, exist_ok=True)

    return program_dir

def get_program_dir_path(proposal_id, visit=None, root_dir='pipeline'):
    """Return the program directory for IPPPSSOOT from a proposal id. If visit is
    specified, return the visit directory.

    Input:
        proposal_id:    a proposal id.
        visit:          the two character designation for the HST visit
        root_dir:       root directory of the program, it's either 'staging', 'pipeline'
                        or 'bundles'.
    """
    root = HST_DIR[root_dir]
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    if visit is None:
        program_dir = root + '/hst_' + formatted_proposal_id
    else:
        program_dir = root + '/hst_' + formatted_proposal_id + f'/visit_{visit}'

    return program_dir

def get_deliverable_path(proposal_id):
    """Return the final deliverable path in the bundles directory for a given proposal
    id.

    Input:
        proposal_id:    a proposal id.
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    return (get_program_dir_path(proposal_id, None, 'bundles') +
            '/hst_' +
            formatted_proposal_id +
            '-deliverable')

def get_format_term(filename):
    """Return IPPPSSOOT for a given file name.

    Input:
        filename:   a product file name
    """
    format_term, _, _ = filename.partition('_')
    return format_term

def get_instrument_id_from_fname(filename):
    """Return instrument id for a given file name.

    Input:
        filename:   a product file name
    """
    letter_code = filename.lower()[0]
    return (INSTRUMENT_FROM_LETTER_CODE[letter_code]
            if letter_code in INSTRUMENT_FROM_LETTER_CODE else None)

def get_file_suffix(filename):
    """Return suffix for a given file name.

    Input:
        filename:   a product file name
    """
    filename, _, _ = filename.rpartition('.')
    _ ,_ , suffix = filename.partition('_')
    return suffix

def get_visit(format_term):
    """Return the two characters of HST visit.

    Input:
        format_term:    the first 8 or 9 characters of the file name (IPPPSSOOT).
    """
    return format_term[4:6]

def file_md5(filepath):
    """Find the hexadecimal digest (checksum) of a file in the filesystem.

    Input:
        filepath:   the path of the targeted file.
    """
    chunk_size = 4096
    hasher = md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def get_formatted_proposal_id(proposal_id):
    """Prepend 0 to the given proposal id if necessary

    Input:
        proposal_id:   the proposal id.
    """
    return str(proposal_id).zfill(5)

def backup_file(proposal_id, visit, filepath):
    """Rename and move a file to the /backups.

    Input:
        proposal_id:    the proposal id.
        visit:          the two character visit.
        filepath:       the current filepath to be renamed & moved.
    """
    backups_dir = get_program_dir_path(proposal_id, visit) + '/backups'
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    _, _, fname = filepath.rpartition('/')
    basename, _, ext = fname.partition('.')
    new_path = f'{backups_dir}/{basename}-{now}.{ext}'
    # create backups dir if it doesn't exist
    os.makedirs(backups_dir, exist_ok=True)
    # move file to the back up dir
    shutil.move(filepath, new_path)
