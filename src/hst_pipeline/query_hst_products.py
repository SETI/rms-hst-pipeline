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
#   - Compare the files with products.txt and trl_checksums.txt in
#     <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/; if they changed, write the new lists as
#     products.txt.pending and trl_checksums.txt.pending, which replace them once the
#     visit has been labeled.
# - Delete all TRL files.
# - Return the tuple of changed visits & all available visists.
##########################################################################################
import os
import pdslogger
import shutil
from collections import defaultdict

from hst_pipeline.hst_helper import (DOWNLOAD_TMP_PREFIX,
                                     PENDING_SUFFIX,
                                     PRODUCTS_FILE,
                                     TRL_CHECKSUMS_FILE)
from hst_pipeline.hst_helper.fs_utils import (create_program_dir,
                                              file_md5,
                                              get_program_dir_path,
                                              get_visit)
from hst_pipeline.hst_helper.query_utils import (download_files,
                                                 get_filtered_products,
                                                 get_trl_products,
                                                 query_mast_slice,
                                                 rename_tmp_prefixed_downloads,
                                                 tmp_download_filename)

# A dictionary keyed by IPPPSSOOT and stores observation id from MAST as the value.
products_obs_dict = {}

def query_hst_products(proposal_id, logger=None):
    """These actions are performed:

        - Query MAST for all available visits and files in this program.
        - Create directories <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/ if they don't exist.
        - Download all TRL files to <HST_STAGING>/hst_<nnnnn>/.
        - Compare with PRODUCTS_FILE & TRL_CHECKSUMS_FILE; write pending versions of
          those that changed.
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
    except ValueError: #pragma: no cover
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

        if product_fname not in files_dict[visit]:
            files_dict[visit].append(product_fname)
        if 'trl' in product_fname and product_fname not in trl_files_dict[visit]:
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
        logger.info(f'Download trl products to {trl_dir}')
        download_files(trl_products, trl_dir, logger,
                       filename_fn=tmp_download_filename)
        rename_tmp_prefixed_downloads(trl_dir, DOWNLOAD_TMP_PREFIX, logger)
    except: #pragma: no cover
        # Downloading failed, removed all the trl files to restore a clean directory.
        # We will only have either all files downloaded or zero file downloaded.
        for f in os.listdir(trl_dir):
            if 'mastDownload' in f:
                dir_path = os.path.join(trl_dir, f)
                shutil.rmtree(dir_path)

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
            try:
                os.remove(filepath)
            except FileNotFoundError: #pragma: no cover
                pass

    return (visit_diff, list(files_dict.keys()))

def compare_files_txt(proposal_id, files_dict, visit, fname, checksum_included=False):
    """Return a flag to indicate if any files are new or changed in the visit since it
    was last labeled.

    Compare the visit's txt file, which describes its files as of its last labeling,
    with the results from MAST. If they differ, or there is no txt file, write the new
    list to "<fname>.pending"; fs_utils.commit_visit_lists() moves it into place once
    the visit has been labeled. If they match, remove any pending file left by an
    earlier run.

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
    txt_file_path = f'{get_program_dir_path(proposal_id, visit)}/{fname}'
    pending_path = txt_file_path + PENDING_SUFFIX
    files_li = sorted(files_dict[visit])
    if checksum_included:
        files_li = [f'{file}:{file_md5(get_downloaded_file_path(proposal_id, file))}'
                    for file in files_li]

    if os.path.exists(txt_file_path):
        with open(txt_file_path) as f:
            files_from_txt = [line.rstrip() for line in f]
        if files_from_txt == files_li:
            if os.path.exists(pending_path):
                os.remove(pending_path)
            return False

    with open(pending_path, 'w') as f:
        for line in files_li:
            f.write(line + '\n')
    return True

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
