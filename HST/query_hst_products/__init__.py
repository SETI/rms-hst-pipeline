##########################################################################################
# query_hst_products/__init__.py
##########################################################################################
import os
import pdslogger

from collections import defaultdict

from hst_helper import (PRODUCTS_FILE,
                        TRL_CHECKSUMS_FILE)
from hst_helper.query_utils import (download_files,
                                    get_filtered_products,
                                    get_trl_products,
                                    query_mast_slice)
from hst_helper.fs_utils import (backup_file,
                                 create_program_dir,
                                 file_md5,
                                 get_downloaded_file_path,
                                 get_program_dir_path,
                                 get_visit)

def query_hst_products(proposal_id, logger=None):
    """Return all accepted products from mast with a given proposal id .
    Input:
        proposal_id:    a proposal id.
    """
    visit_diff = []
    logger = logger or pdslogger.EasyLogger()

    logger.info('Query hst products for propsal id: ', str(proposal_id))
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Query mast
    table = query_mast_slice(proposal_id=proposal_id, logger=logger)
    filtered_products = get_filtered_products(table)

    # Log all accepted file names
    logger.info(f'List out all accepted files from mast for {proposal_id}')
    files_dict = defaultdict(list)
    trl_files_dict = defaultdict(list)
    for row in filtered_products:
        productFilename = row['productFilename']
        format_term, _, _ = productFilename.partition('_')
        visit = get_visit(format_term)

        files_dict[visit].append(productFilename)
        if 'trl' in productFilename:
            trl_files_dict[visit].append(productFilename)

        suffix = row['productSubGroupDescription']
        logger.info(f'File: {productFilename} with suffix: {suffix}')

     # Create program and all visits directories
    logger.info('Create program and visit directories that do not already exist.')
    for visit in files_dict:
        create_program_dir(proposal_id, visit)

    # Download all TRL files
    trl_dir = create_program_dir(proposal_id=proposal_id, root_dir='staging')
    logger.info(f'Download all TRL files for {proposal_id} to {trl_dir}')
    trl_products = get_trl_products(table)
    download_files(trl_products, trl_dir, logger)

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

    return visit_diff

def generate_files_txt(proposal_id, files_dict, visit, fname, checksum_included=False):
    """Create {fname}.txt in the pipeline visit directory of a proposal id. The file
    will contain a list of file names of the visit in alphabetical order.
    if checksum_included is True, the file will contain a dictionary keyed by file names
    and the corresponding checksums as the values.
    Input:
        proposal_id:        a proposal id.
        files_dict:         a dictionary keyed by two character visit and store a list of
                            files for the corresponding visit.
        visit:              two character visit.
        fname:              the file name.
        checksum_included:  a flag used to deteremine if we want to include checksum
                            of each file in the generated file.

    """
    file_path = get_program_dir_path(proposal_id, visit) + f'/{fname}'
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
    """Return a flag to indicate whether any files are new or changed in the visit.
    Compare the contents of current txt file with the results from Mast. If they are
    the same, keep the current txt file. If they are different, move the current txt
    file to the backups directory, and generate the new txt file based on the new results
    from Mast.
    Input:
        proposal_id:        a proposal id.
        files_dict:         a dictionary keyed by two character visit and store a list of
                            files for the corresponding visit.
        visit:              two character visit.
        fname:              the file name.
        checksum_included:  a flag used to deteremine if we want to include checksum
                            of each file in the generated file.

    """
    is_visit_diff = False
    txt_file_path = get_program_dir_path(proposal_id, visit) + f'/{fname}'
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

        # compare the list from Mast with the contents of the old txt file
        if files_from_txt != files_li:
            is_visit_diff = True
            # move the old txt file to backups
            backup_file(proposal_id, visit, txt_file_path)
            # generate the new txt file
            generate_files_txt(proposal_id, files_dict, visit,
                               f'{fname}', checksum_included)
    else:
        # Generate txt file if it doesn't exist
        generate_files_txt(proposal_id, files_dict, visit, f'{fname}', checksum_included)
        is_visit_diff = True

    return is_visit_diff