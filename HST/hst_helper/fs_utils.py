##########################################################################################
# hst_helper/fs_utils.py
##########################################################################################
import os

from hashlib import md5

from . import HST_DIR

def create_program_dir(proposal_id, visit=None, root_dir="pipeline"):
    """Create the program directory for IPPPSSOOT from a proposal id, return the path of
    the directory. If visit is specified, create the visit directory as well.
    Input:
        proposal_id:    a proposal id.
        visit:          the two character designation for the HST visit
        root_dir:       root directory of the program, it's either "staging", "pipeline"
                        or "bundles".
    """
    program_dir = get_program_dir_path(proposal_id, visit, root_dir)

    if not os.path.isdir(program_dir):
        os.makedirs(program_dir)
    return program_dir

def get_program_dir_path(proposal_id, visit=None, root_dir="pipeline"):
    """Return the program directory for IPPPSSOOT from a proposal id. If visit is
    specified, return the visit directory.
    Input:
        proposal_id:    a proposal id.
        visit:          the two character designation for the HST visit
        root_dir:       root directory of the program, it's either "staging", "pipeline"
                        or "bundles".
    """
    root = HST_DIR[root_dir]
    if visit is None:
        program_dir = root + "/hst_" + str(proposal_id).zfill(5)
    else:
        program_dir = root + "/hst_" + str(proposal_id).zfill(5) + f"/visit_{visit}"

    return program_dir

# def is_missing_program_dir(proposal_id, root_dir="pipeline"):
#     """Check if a program directory for a proposal id is missing
#     Input:
#         proposal_id:    a proposal id.
#         root_dir:       root directory of the program, it's either "staging", "pipeline"
#                         or "bundles".
#     """
#     root = HST_DIR[root_dir]
#     program_dir = root + "/hst_" + str(proposal_id).zfill(5)
#     return os.path.isdir(program_dir)

def get_format_term(filename):
    """Return IPPPSSOOT for a given file name.
    Input:
        filename:   a product file name
    """
    format_term, _, _ = filename.partition("_")
    return format_term

def get_visit(format_term):
    """Return the two characters of HST visit
    Input:
        format_term:    the first 8 or 9 characters of the file name (IPPPSSOOT).
    """
    return format_term[4:6]

def construct_downloaded_file_path(proposal_id, fname, visit=None, root_dir="staging"):
    """Return the file path of a downloaded file.
    Input:
        proposal_id:    a proposal id.
        fname:          the file name.
        visit:          two character visit if the file is stored under visit dir.
        root_dir:       the root directory of the store file.

    """
    return (get_program_dir_path(proposal_id, visit, root_dir) +
            "/mastDownload/HST/" +
            get_format_term(fname) +
            f"/{fname}")

def file_md5(filepath):
    """Find the hexadecimal digest (checksum) of a file in the filesystem.
    Input:
        filepath:   the path of the targeted file
    """
    CHUNK = 4096
    hasher = md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
