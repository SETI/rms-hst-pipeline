##########################################################################################
# query_mast/__init__.py
##########################################################################################

import datetime
import time
import filecmp
import fnmatch
import os
import re

import pdslogger
import julian

from astropy.table import Table
from astropy.table.row import Row
from astroquery.mast import Observations
from requests.exceptions import ConnectionError

from product_labels.suffix_info import (ACCEPTED_SUFFIXES,
                                        ACCEPTED_LETTER_CODES, 
                                        INSTRUMENT_FROM_LETTER_CODE)


TWD = os.environ["TMP_WORKING_DIR"]
DEFAULT_DIR = TWD + "/files_from_mast"
START_DATE = (1900, 1, 1)
END_DATE = (2025, 1, 1)

def ymd_tuple_to_mjd(ymd):
    """Return Modified Julian Date.
    Input:
        ymd:    a tuple of year, month, and day.
    """
    y, m, d = ymd
    days = julian.day_from_ymd(y, m, d)
    return julian.mjd_from_day(days)

def filter_table(row_predicate, table):
    """Return a copy of the filtered table object based on the return of row_predicate.
    Input:
        row_predicate:    a function with the condition used to filter the table.
        table:            target table to be filtered.
    """
    to_delete = [n for (n, row) in enumerate(table) if not row_predicate(row)]
    copy = table.copy()
    copy.remove_rows(to_delete)
    return copy

def get_products_from_mast(proposal_id,
                           start_date=START_DATE,
                           end_date=END_DATE,
                           logger=None,
                           max_retries=1,
                           dir=DEFAULT_DIR):
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
                             max_retries)
    filtered_products = get_filtered_products(table)
    # Log all accepted file names
    logger.info(f"List out all accepted files from mast for {proposal_id}")
    for row in filtered_products:
        productFilename = row["productFilename"]
        suffix = row["productSubGroupDescription"]
        logger.info(f"File: {productFilename} with suffix: {suffix}")
    # Download all accepted files
    download_files(filtered_products, proposal_id, logger, dir)


def query_mast_slice(proposal_id,
                     start_date=START_DATE,
                     end_date=END_DATE,
                     logger=None,
                     max_retries=1):
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
            table = Observations.query_criteria(
                dataRights="PUBLIC",
                obs_collection=["HST"],
                proposal_id=str(proposal_id),
                t_obs_release=(start_date, end_date),
                mtFlag=True
            )
            return table
            # return get_products(table)
        except ConnectionError as e:
            retry = retry + 1
            logger.info(f"retry #{retry}: {e}")
            time.sleep(1)
    logger.exception(RuntimeError)
    raise RuntimeError("Query mast timed out. Number of retries: " + max_retries)

def download_files(table, proposal_id, logger=None, dir=DEFAULT_DIR):
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
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)

    if len(table) > 0:
        logger.info("Download files to " + working_dir)
        Observations.download_products(table, download_dir=working_dir)

def get_filtered_products(table):
    """Return product rows of an observation table with accepted instrument letter code
    and suffxes.
    Input:
        table:  an observation table from mast query.
    """
    result = Observations.get_product_list(table)
    result = filter_table(is_accepted_instrument_letter_code, result)
    result = filter_table(is_accepted_instrument_suffix, result)
    # print(ACCEPTED_SUFFIXES)
    return result


def is_accepted_instrument_letter_code(row):
    """Check if a product row has accepted letter code in the first letter of the 
    product filename.
    Input:
        row:    an observation table row.
    """
    return row["obs_id"][0].lower() in ACCEPTED_LETTER_CODES

def is_accepted_instrument_suffix(row):
    """Check if a product row has accepted suffex in the productSubGroupDescription field
    of the table.
    Input:
        row:    an observation table row.
    """
    suffix = get_suffix(row)
    instrument_id = get_instrument_id(row)
    # For files like n4wl03fxq_raw.jpg with "--" will raise an error
    # return is_accepted(suffix, instrument_id) 
    return suffix in ACCEPTED_SUFFIXES[instrument_id]

def get_instrument_id(row):
    """Return the instrument id for a given product row.
    Input:
        row:    an observation table row.
    """
    return INSTRUMENT_FROM_LETTER_CODE[row["obs_id"][0].lower()]

def get_suffix(row):
    """Return the product file suffix for a given product row.
    Input:
        row:    an observation table row.
    """
    return str(row["productSubGroupDescription"]).lower()