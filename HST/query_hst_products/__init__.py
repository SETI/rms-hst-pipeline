##########################################################################################
# query_hst_products/__init__.py
##########################################################################################
import pdslogger

from collections import defaultdict
from hst_helper import (START_DATE,
                        END_DATE,
                        RETRY,
                        HST_DIR)
from hst_helper.query_utils import (download_files,
                                     get_filtered_products,
                                     get_trl_products,
                                     query_mast_slice)
from hst_helper.fs_utils import (create_program_dir,
                                 file_md5,
                                 construct_downloaded_file_path,
                                 get_format_term,
                                 get_program_dir_path,
                                 get_visit)

def query_hst_products(proposal_id, logger):
    """Return all accepted products from mast with a given proposal id .
    Input:
        proposal_id:    a proposal id.
    """
    logger = logger or pdslogger.EasyLogger()

    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f"Proposal id: {proposal_id} is not valid.")

    logger.info("Query hst products for propsal id: ", str(proposal_id))
    # Query mast
    table = query_mast_slice(proposal_id=proposal_id, logger=logger)
    filtered_products = get_filtered_products(table)

    # Log all accepted file names
    logger.info(f"List out all accepted files from mast for {proposal_id}")
    files_dict = defaultdict(list)
    trl_files_dict = defaultdict(list)
    for row in filtered_products:
        productFilename = row["productFilename"]
        format_term, _, _ = productFilename.partition("_")
        visit = get_visit(format_term)

        files_dict[visit].append(productFilename)
        if "trl" in productFilename:
            trl_files_dict[visit].append(productFilename)

        suffix = row["productSubGroupDescription"]
        logger.info(f"File: {productFilename} with suffix: {suffix}")

    # create program and all visits directories
    logger.info("Create program and visit directories that do not already exist.")
    for visit in files_dict.keys():
        create_program_dir(proposal_id=proposal_id, visit=visit)

    # Generate products.txt for different visits
    logger.info("Create products.txt that do not already exist.")
    generate_files_txt(proposal_id, files_dict, "products.txt")


    # download all TRL files
    logger.info("Download trl files.")
    trl_products = get_trl_products(table)

    # Generate trl_checksums.txt for different visits
    logger.info("Create trl_checksums.txt that do not already exist.")
    generate_files_txt(proposal_id, trl_files_dict, "trl_checksums.txt", True)

    files_dir = create_program_dir(proposal_id=proposal_id, root_dir="staging")
    logger.info(f"Download all TRL files for {proposal_id} to {files_dir}")
    download_files(trl_products, files_dir, logger)

    return

def generate_files_txt(proposal_id, files_dict, fname, checksum_included=False):
    """Create {fname}.txt in the pipeline visit directory of a proposal id. The file
    will contain a list of file names of the visit in alphabetical order.
    if checksum_included is True, the file will contain a dictionary keyed by file names
    and the corresponding checksums as the values.
    Input:
        proposal_id:        a proposal id.
        files_dict:         a dictionary keyed by two character visit and store a list of
                            files for the corresponding visit.
        fname:              the file name.
        checksum_included:  a flag used to deteremine if we want to include checksum
                            of each file in the generated file.

    """
    for visit in files_dict:
        files_dict[visit].sort()
        file_path = get_program_dir_path(proposal_id, visit) + f"/{fname}"
        with open(file_path, "w") as f:
            for file in files_dict[visit]:
                if not checksum_included:
                    f.write("%s\n" % file)
                else:
                    filepath = construct_downloaded_file_path(proposal_id, file)
                    checksum = file_md5(filepath)
                    f.write("%s:%s\n" % (file, checksum))
