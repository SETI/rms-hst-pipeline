##########################################################################################
# hst_helper/query_utils.py
##########################################################################################
import os

import julian
import pdslogger

from astroquery.mast import Observations

from . import (START_DATE,
               END_DATE,
               RETRY)

from product_labels.suffix_info import (ACCEPTED_SUFFIXES,
                                        ACCEPTED_LETTER_CODES,
                                        INSTRUMENT_FROM_LETTER_CODE)

def ymd_tuple_to_mjd(ymd):
    """Return Modified Julian Date.
    Input:
        ymd:    a tuple of year, month, and day.
    """
    y, m, d = ymd
    days = julian.day_from_ymd(y, m, d)
    return julian.mjd_from_day(days)

def query_mast_slice(proposal_id=None,
                     instrument=None,
                     start_date=START_DATE,
                     end_date=END_DATE,
                     logger=None,
                     max_retries=RETRY,
                     testing=False):
    """Return a slice of mast database as a table object with a given proposal id,
    instrument, start_date, and end_date.
    Input:
        proposal_id:    a proposal id.
        instrument:     a instrument name.
        start_date:     observation start datetime.
        end_date:       observation end datetime.
        logger:         pdslogger to use; None for default EasyLogger.
        max_retries:    number of retries when there is a connection to mast.
    """
    logger = logger or pdslogger.EasyLogger()
    start_date = ymd_tuple_to_mjd(start_date)
    end_date = ymd_tuple_to_mjd(end_date)
    retry = 0
    logger.info("Query MAST: run query_mast_slice")

    query_params = {
        "dataRights": "PUBLIC",
        "obs_collection": ["HST"],
        # "t_obs_release": (start_date, end_date),
        "mtFlag": True
    }

    if proposal_id is not None:
        query_params["proposal_id"] = str(proposal_id)
    if instrument is not None:
        query_params["instrument_name"] = str(instrument)
    if start_date is not None and end_date is not None:
        query_params["t_obs_release"] = (start_date, end_date)

    for retry in range(max_retries):
        try:
            if testing and max_retries > 1:
                raise ConnectionError
            table = Observations.query_criteria(**query_params)
            # table = Observations.query_criteria(
            #     dataRights="PUBLIC",
            #     obs_collection=["HST"],
            #     proposal_id=str(proposal_id),
            #     t_obs_release=(start_date, end_date),
            #     mtFlag=True
            # )
            return table
        except ConnectionError as e:
            retry = retry + 1
            logger.info(f"retry #{retry}: {e}")
            time.sleep(1)
    logger.exception(RuntimeError)
    raise RuntimeError("Query mast timed out. Number of retries: " + str(max_retries))

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

def is_trl_suffix(row):
    """Check if a product row has trl in the productSubGroupDescription field
    of the table.
    Input:
        row:    an observation table row.
    """
    return get_suffix(row) == 'trl'

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

def get_trl_products(table):
    """Return product rows of an observation table with trl suffix
    Input:
        table:  an observation table from mast query.
    """
    result = Observations.get_product_list(table)
    result = filter_table(is_accepted_instrument_letter_code, result)
    result = filter_table(is_trl_suffix, result)
    return result

def download_files(table, dir, logger=None, testing=False):
    """Download files from mast for a given product table and proposal id
    Input:
        table:          an observation table from mast query.
        proposal_id:    a proposal id.
        dir:            the directory we want to store the downloaded files.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    if not os.path.isdir(dir): # pragma: no cover
        os.makedirs(dir)

    if len(table) > 0:
        logger.info("Download files to " + dir)
        if not testing: # pragma: no cover, no need to download files during the test
            Observations.download_products(table, download_dir=dir)