##########################################################################################
# query_hst_products.py
#
# query_hst_products is the main function called in query_hst_products pipeline task
# script. It will do these actions:
#
# - Query MAST to get all available visits and files in this program.
# - Create directories <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/ if they don't exist.
# - Download all TRL files to <HST_STAGING>/hst_<nnnnn>/.
# - For each visit:
#   - Update or create products.txt in <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/.
#   - Update or create trl_checksums.txt in <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/.
# - Delete all TRL files.
# - Return the tuple of changed visits & all available visists.
##########################################################################################

import os
import pdslogger
import shutil
from collections import defaultdict

from hst_helper import (PRODUCTS_FILE,
                        TRL_CHECKSUMS_FILE)
from hst_helper.fs_utils import (backup_file,
                                 create_program_dir,
                                 file_md5,
                                 get_formatted_proposal_id,
                                 get_program_dir_path,
                                 get_visit)
from hst_helper.query_utils import (download_files,
                                    get_filtered_products,
                                    get_trl_products,
                                    query_mast_slice)
from queue_manager.task_queue_db import remove_all_tasks_for_a_prog_id

# A dictionary keyed by IPPPSSOOT and stores observation id from MAST as the value.
products_obs_dict = {}

def query_hst_products(proposal_id, logger=None):
    """These actions are performed:

        - Query MAST for all available visits and files in this program.
        - Create directories <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/ if they don't exist.
        - Download all TRL files to <HST_STAGING>/hst_<nnnnn>/.
        - Compare and create PRODUCTS_FILE & TRL_CHECKSUMS_FILE.
        - Delete all TRL files.
        - Return the tuple of changed visits & all available visists.

    Input:
        proposal_id    a proposal id.

    Returns:    a tuple of a list of visits in which any files are new or changed and a
                list of all visits for the given proposal id.
    """
    visit_diff = []
    logger = logger or pdslogger.EasyLogger()

    logger.info('Query hst products for propsal id: ', str(proposal_id))
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Query MAST for all available visits and files in this program
    table = query_mast_slice(proposal_id=proposal_id, logger=logger)
    filtered_products = get_filtered_products(table)
    # Log all accepted file names
    logger.info(f'List out all accepted files from MAST for {proposal_id}')
    files_dict = defaultdict(list)
    trl_files_dict = defaultdict(list)
    for row in filtered_products:
        product_fname = row['productFilename']
        obs_id = row['obs_id']
        products_obs_dict[product_fname] = obs_id
        format_term, _, _ = product_fname.partition('_')
        visit = get_visit(format_term)

        files_dict[visit].append(product_fname)
        if 'trl' in product_fname:
            trl_files_dict[visit].append(product_fname)

        suffix = row['productSubGroupDescription']
        logger.info(f'File: {product_fname} with suffix: {suffix}')

    # Create program and all visits directories if they don't exist.
    logger.info('Create program and visit directories that do not already exist')
    for visit in files_dict:
        create_program_dir(proposal_id, visit)

    # Download all TRL files
    trl_dir = create_program_dir(proposal_id=proposal_id, root_dir='staging')
    logger.info(f'Download all TRL files for {proposal_id} to {trl_dir}')
    trl_products = get_trl_products(table)

    try:
        download_files(trl_products, trl_dir, logger)
    except:
        # Downloading failed, removed all the trl files to restore a clean directory.
        # We will only have either all files downloaded or zero file downloaded.
        for f in os.listdir(trl_dir):
            if 'mastDownload' in f:
                dir_path = os.path.join(trl_dir, f)
                shutil.rmtree(dir_path)

        # Before raising the error, remove the task queue & subprocess of the proposal id
        # from database.
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)
        remove_all_tasks_for_a_prog_id(formatted_proposal_id)

        logger.exception('MAST trl files downlaod failure')
        raise

    # Compare and create PRODUCTS_FILE & TRL_CHECKSUMS_FILE
    logger.info(f'Create {PRODUCTS_FILE} and {TRL_CHECKSUMS_FILE}')
    for visit in files_dict:
        logger.info(f'Create {PRODUCTS_FILE} for visit: {visit} of {proposal_id}')
        prod_diff = compare_files_txt(proposal_id, files_dict,
                                      visit, PRODUCTS_FILE)
        logger.info(f'Create {TRL_CHECKSUMS_FILE} for visit: {visit} of {proposal_id}')
        trl_diff = compare_files_txt(proposal_id, trl_files_dict,
                                     visit, TRL_CHECKSUMS_FILE, True)
        if prod_diff or trl_diff:
            visit_diff.append(visit)

    # Delete all TRL files
    logger.info(f'Delete all TRL files for {proposal_id} in {trl_dir}')
    for visit, trl_files in trl_files_dict.items():
        for f in trl_files:
            filepath = get_downloaded_file_path(proposal_id, f)
            os.remove(filepath)

    # Clean up the empty mastDownload directory
    staging_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
    shutil.rmtree(staging_dir + '/mastDownload')

    return (visit_diff, list(files_dict.keys()))

def generate_files_txt(proposal_id, files_dict, visit, fname, checksum_included=False):
    """Create {fname}.txt in the pipeline visit directory of a proposal id. The file
    will contain a list of file names of the visit in alphabetical order.
    if checksum_included is True, the file will contain a dictionary keyed by file names
    and the corresponding checksums as the values.

    Input:
        proposal_id          a proposal id.
        files_dict           a dictionary keyed by two character visit and store a list
                             of files for the corresponding visit.
        visit                two character visit.
        fname                the file name.
        checksum_included    a flag used to deteremine if we want to include checksum
                             of each file in the generated file.

    """
    file_path = f'{get_program_dir_path(proposal_id, visit)}/{fname}'
    files_li = files_dict[visit]

    with open(file_path, 'w') as f:
        for file in files_li:
            if not checksum_included:
                f.write('%s\n' % file)
            else:
                filepath = get_downloaded_file_path(proposal_id, file)
                checksum = file_md5(filepath)
                f.write('%s:%s\n' % (file, checksum))

def compare_files_txt(proposal_id, files_dict, visit, fname, checksum_included=False):
    """Return a flag to indicate if any files are new or changed in the visit.
    Compare the contents of current txt file with the results from MAST. If they are
    the same, keep the current txt file. If they are different, move the current txt
    file to the backups directory, and generate the new txt file based on the new results
    from MAST.

    Input:
        proposal_id          a proposal id.
        files_dict           a dictionary keyed by two character visit and store a list
                             of files for the corresponding visit.
        visit                two character visit.
        fname                the file name.
        checksum_included    a flag used to deteremine if we want to include checksum
                             of each file in the generated file.

    Returns:    a boolean that indicates if any files are new or changed in the visit.
    """
    is_visit_diff = False
    txt_file_path = f'{get_program_dir_path(proposal_id, visit)}/{fname}'
    files_li = files_dict[visit]
    files_li.sort()
    if checksum_included:
        files_li_with_checksum = []
        for file in files_li:
            filepath = get_downloaded_file_path(proposal_id, file)
            checksum = file_md5(filepath)
            files_li_with_checksum.append(f'{file}:{checksum}')
        files_li = files_li_with_checksum

    if os.path.exists(txt_file_path):
        # get info from the old txt file
        text_file = open(txt_file_path, 'r')
        files_from_txt = [line.rstrip() for line in text_file]
        text_file.close()

        # compare the list from MAST with the contents of the old txt file
        if files_from_txt != files_li:
            is_visit_diff = True
            # move the old txt file to backups
            backup_file(proposal_id, visit, txt_file_path)
            # generate the new txt file
            generate_files_txt(proposal_id, files_dict, visit,
                               fname, checksum_included)
    else:
        # Generate txt file if it doesn't exist
        generate_files_txt(proposal_id, files_dict, visit, fname, checksum_included)
        is_visit_diff = True

    return is_visit_diff

def get_downloaded_file_path(proposal_id, fname, visit=None, root_dir='staging'):
    """Return the file path of a downloaded file.

    Input:
        proposal_id    a proposal id.
        fname          the file name.
        visit          two character visit if the file is stored under visit dir.
        root_dir       the root directory of the store file.

    Returns:    the file path of a downloaded file.
    """
    return (f'{get_program_dir_path(proposal_id, visit, root_dir)}/mastDownload/HST/'
            f'{products_obs_dict[fname]}/{fname}')
