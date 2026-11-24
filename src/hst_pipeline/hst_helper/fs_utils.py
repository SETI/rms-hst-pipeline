##########################################################################################
# hst_helper/fs_utils.py
#
# This file contains helper functions related file system, includig creating directories,
# getting program directory path, get file suffix, get file checksum, and etc.
##########################################################################################

import datetime
from hashlib import md5
import os
import shutil

from . import DELIVERABLE_SUFFIX, HST_DIR, PENDING_SUFFIX, VISIT_LISTS
from hst_pipeline.product_labels.suffix_info import (INSTRUMENT_FROM_LETTER_CODE,
                                                     product_id)

def create_program_dir(proposal_id, visit=None, root_dir='pipeline'):
    """Create the program directory for IPPPSSOOT from a proposal id, return the path of
    the directory. If visit is specified, create the visit directory as well.

    Input:
        proposal_id    a proposal id.
        visit          the two character designation for the HST visit
        root_dir       root directory of the program, it's either 'staging', 'pipeline'
                       or 'bundles'.

    Returns:    the path of the newly created program directory.
    """
    program_dir = get_program_dir_path(proposal_id, visit, root_dir)
    os.makedirs(program_dir, exist_ok=True)

    return program_dir

def get_program_dir_path(proposal_id, visit=None, root_dir='pipeline', testing=False):
    """Return the program directory for IPPPSSOOT from a proposal id. If visit is
    specified, return the visit directory.

    Under the bundles root, the program directory is named hst_<nnnnn>-deliverable; it
    holds the bundle directory hst_<nnnnn> (see get_deliverable_path) together with the
    manifest files and the validation report.

    Input:
        proposal_id    a proposal id.
        visit          the two character designation for the HST visit
        root_dir       root directory of the program, it's either 'staging', 'pipeline'
                       or 'bundles'.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.

    Returns:    the path of the newly created program directory.
    """
    root = HST_DIR[root_dir]
    # Create separate directories for testing. Tests will setup and tear down the testing
    # dirctories so the regular directories will remain intact.
    if not testing:
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    else:
        formatted_proposal_id = f'{get_formatted_proposal_id(proposal_id)}_testing'

    program_dir = f'{root}/hst_{formatted_proposal_id}'
    if root_dir == 'bundles':
        program_dir += DELIVERABLE_SUFFIX
    if visit is not None:
        program_dir += f'/visit_{visit}'

    return program_dir

def get_deliverable_path(proposal_id, testing=False):
    """Return the path of the bundle itself for a given proposal id.

    Input:
        proposal_id    a proposal id.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.

    Returns:    <HST_BUNDLES>/hst_<nnnnn>-deliverable/hst_<nnnnn>, the directory holding
                bundle.lblx and the collection directories.
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    return (f'{get_program_dir_path(proposal_id, None, "bundles", testing)}'
            f'/hst_{formatted_proposal_id}')

def get_context_products_path(proposal_id, testing=False):
    """Return the path of the temporary context-products JSON file that is passed to the
    validator for a given proposal id. It sits beside the bundle, in the program
    directory under the bundles root.

    Input:
        proposal_id    a proposal id.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.

    Returns:    <HST_BUNDLES>/hst_<nnnnn>-deliverable/tmp-context-products-<nnnnn>.json
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    return (f'{get_program_dir_path(proposal_id, None, "bundles", testing)}'
            f'/tmp-context-products-{formatted_proposal_id}.json')

def get_format_term(filename):
    """Return IPPPSSOOT for a given file name.

    Input:
        filename   a product file name.

    Returns:    the IPPPSSOOT.
    """
    # Strip extension first so labels like z2no0801t.lblx still yield z2no0801t.
    stem, _, _ = filename.rpartition('.')
    if not stem:
        stem = filename
    format_term, _, _ = stem.partition('_')
    return format_term

def get_product_id(filename):
    """Return the product ID (the last field of the LID) for a given file name.

    Input:
        filename   a product file name, e.g., a FITS file, label, or browse image.

    Returns:    the IPPPSSOOT plus the tail of an extended suffix, e.g., "ld8x010101" for
                "ld8x01010_x1dsum1.fits" and "ld8x01010" for "ld8x01010_x1dsum.png".
    """
    return product_id(get_format_term(filename), get_file_suffix(filename))

def get_instrument_id_from_fname(filename):
    """Return instrument id for a given file name.

    Input:
        filename   a product file name.

    Returns:    the instrument id or None.
    """
    letter_code = filename.lower()[0]
    return (INSTRUMENT_FROM_LETTER_CODE[letter_code]
            if letter_code in INSTRUMENT_FROM_LETTER_CODE else None)

def get_file_suffix(filename):
    """Return suffix for a given file name.

    Input:
        filename    a product file name

    Returns:    the suffix of the given file.
    """
    filename, _, _ = filename.rpartition('.')
    _ ,_ , suffix = filename.partition('_')
    return suffix

def get_visit(format_term):
    """Return the two characters of HST visit.

    Input:
        format_term    the first 8 or 9 characters of the file name (IPPPSSOOT).

    Returns:    the two-character visit code.
    """
    return format_term[4:6]

def file_md5(filepath):
    """Find the hexadecimal digest (checksum) of a file in the filesystem.

    Input:
        filepath    the path of the targeted file.

    Returns:    the checksum of the given file.
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
        proposal_id    the proposal id.

    Returns:    the five-digit proposal id with prepended zero's if the given proposal
                id has less than five digits.
    """
    return str(proposal_id).zfill(5)

# Format of the "-<ymdhms>" that backup_file appends to a file's basename
BACKUP_TIME_FORMAT = '%Y-%m-%dT%H-%M-%S'
BACKUP_TIME_PATTERN = r'\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d'

def backup_time():
    """Return the current time in BACKUP_TIME_FORMAT."""
    return datetime.datetime.now().strftime(BACKUP_TIME_FORMAT)

def backup_file(proposal_id, visit, filepath, now=None):
    """Rename and move a file to the /backups.

    Input:
        proposal_id    the proposal id.
        visit          the two character visit.
        filepath       the current filepath to be renamed & moved.
        now            the time to append, as from backup_time(); None for the current
                       time. Files replaced together share one time.
    """
    backups_dir = f'{get_program_dir_path(proposal_id, visit)}/backups'
    now = now or backup_time()
    _, _, fname = filepath.rpartition('/')
    basename, _, ext = fname.partition('.')
    new_path = f'{backups_dir}/{basename}-{now}.{ext}'
    # create backups dir if it doesn't exist
    os.makedirs(backups_dir, exist_ok=True)
    # move file to the back up dir
    shutil.move(filepath, new_path)

def commit_visit_lists(proposal_id, visit):
    """Replace a visit's PRODUCTS_FILE and TRL_CHECKSUMS_FILE with the pending versions
    written by hst-query-products, backing up the old ones. Call it once the visit has
    been labeled.

    Input:
        proposal_id    the proposal id.
        visit          the two character visit.

    Returns:    the list of the paths that were replaced or created.
    """
    visit_dir = get_program_dir_path(proposal_id, visit)
    committed = []
    now = backup_time()
    for fname in VISIT_LISTS:
        path = f'{visit_dir}/{fname}'
        if not os.path.exists(path + PENDING_SUFFIX):
            continue
        if os.path.exists(path):
            backup_file(proposal_id, visit, path, now=now)
        os.replace(path + PENDING_SUFFIX, path)
        committed.append(path)
    return committed

def discard_pending_visit_lists(proposal_id, visit):
    """Remove a visit's pending PRODUCTS_FILE and TRL_CHECKSUMS_FILE, e.g., after a
    failed retrieval, so that the next query finds the visit changed again.

    Input:
        proposal_id    the proposal id.
        visit          the two character visit.
    """
    visit_dir = get_program_dir_path(proposal_id, visit)
    for fname in VISIT_LISTS:
        try:
            os.remove(f'{visit_dir}/{fname}{PENDING_SUFFIX}')
        except FileNotFoundError:
            pass

def create_col_dir_in_bundle(proposal_id, collection_name, testing=False):
    """Create the collection directory in the final bundle directory for a given propsal
    id & collection name. Return a tuple of the path of the final bundle & collection
    directories.

    Input:
        proposal_id        the proposal id.
        collection_name    the collection name for the directory.
        testing            the flag used to determine if we are calling the function for
                           testing purpose with the test directory.

    Returns:    a tuple of the path of the final bundle & collection directories.
    """
    deliverable_path = get_deliverable_path(proposal_id, testing)
    col_dir = f'{deliverable_path}/{collection_name}'
    os.makedirs(col_dir, exist_ok=True)

    return (deliverable_path, col_dir)
