##########################################################################################
# query_mast/__init__.py
##########################################################################################

import time
import os

import pdslogger

from astroquery.mast import Observations
from requests.exceptions import ConnectionError

from .utils import (filter_table, 
                    is_accepted_instrument_letter_code,
                    is_accepted_instrument_suffix, 
                    ymd_tuple_to_mjd)


TWD = os.environ["TMP_WORKING_DIR"]
DEFAULT_DIR = TWD + "/files_from_mast"
START_DATE = (1900, 1, 1)
END_DATE = (2025, 1, 1)

def get_products_from_mast(proposal_id,
                           start_date=START_DATE,
                           end_date=END_DATE,
                           logger=None,
                           max_retries=1,
                           dir=DEFAULT_DIR,
                           testing=False):
    """Download products from mast with a given proposal id .
    Input:
        proposal_id:    a proposal id.
        start_date:     observation start datetime.
        end_date:       observation end datetime.
        logger:         pdslogger to use; None for default EasyLogger.
        max_retries:    number of retries when there is a connection to mast.
        dir:            the directory to store downloaded files from mast.
    """
    logger = logger or pdslogger.EasyLogger()
    if isinstance(proposal_id, list):
        proposal_id = proposal_id[0]
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f"Proposal id: {proposal_id} is not valid.")
    
    logger.info("Propsal id: ", str(proposal_id))
    # Query mast 
    table = query_mast_slice(proposal_id,
                             start_date,
                             end_date,
                             logger,
                             max_retries,
                             testing)
    filtered_products = get_filtered_products(table)
    # Log all accepted file names
    logger.info(f"List out all accepted files from mast for {proposal_id}")
    for row in filtered_products:
        productFilename = row["productFilename"]
        suffix = row["productSubGroupDescription"]
        logger.info(f"File: {productFilename} with suffix: {suffix}")
    # Download all accepted files
    download_files(filtered_products, proposal_id, logger, dir, testing)

def query_mast_slice(proposal_id,
                     start_date=START_DATE,
                     end_date=END_DATE,
                     logger=None,
                     max_retries=1,
                     testing=False):
    """Return a slice of mast database as a table object with a given proposal id,
    start_date, and end_date.
    Input:
        proposal_id:    a proposal id.
        start_date:     observation start datetime.
        end_date:       observation end datetime.
        logger:         pdslogger to use; None for default EasyLogger.
        max_retries:    number of retries when there is a connection to mast.
    """
    logger = logger or pdslogger.EasyLogger()
    start_date = ymd_tuple_to_mjd(start_date)
    end_date = ymd_tuple_to_mjd(end_date)
    retry = 0
    logger.info("Query mast")
    for retry in range(max_retries):
        try:
            if testing and max_retries > 1:
                raise ConnectionError
            table = Observations.query_criteria(
                dataRights="PUBLIC",
                obs_collection=["HST"],
                proposal_id=str(proposal_id),
                t_obs_release=(start_date, end_date),
                mtFlag=True
            )
            return table
        except ConnectionError as e:
            retry = retry + 1
            logger.info(f"retry #{retry}: {e}")
            time.sleep(1)
    logger.exception(RuntimeError) 
    raise RuntimeError("Query mast timed out. Number of retries: " + str(max_retries))

def get_filtered_products(table):
    """Return product rows of an observation table with accepted instrument letter code
    and suffxes.
    Input:
        table:  an observation table from mast query.
    """
    result = Observations.get_product_list(table)
    result = filter_table(is_accepted_instrument_letter_code, result)
    result = filter_table(is_accepted_instrument_suffix, result)
    return result

def download_files(table, proposal_id, logger=None, dir=DEFAULT_DIR, testing=False):
    """Download files from mast for a given product table and proposal id
    Input:
        table:          an observation table from mast query.
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
        dir:            the directory we want to store the downloaded files. Default to
                        /files_from_mast under TMP_WORKING_DIR in the local environment.  
    """
    logger = logger or pdslogger.EasyLogger()
    hst_segment = f"hst_{proposal_id:05}"
    working_dir = os.path.join(dir, hst_segment)
    if not os.path.isdir(working_dir): # pragma: no cover
        os.makedirs(working_dir)

    if len(table) > 0:
        logger.info("Download files to " + working_dir)
        if not testing: # pragma: no cover, no need to download files during the test
            Observations.download_products(table, download_dir=working_dir) 