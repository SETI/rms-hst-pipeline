##########################################################################################
# hst_helper/query_utils.py
#
# This file contains helper functions related to MAST query, including querying MAST,
# getting file suffix & instrument ids from the table row, downloading files, and etc.
##########################################################################################

import julian
import os
import pdslogger
import time

from astroquery.mast import Observations
from . import (START_DATE,
               END_DATE,
               RETRY)
from .fs_utils import (get_formatted_proposal_id,
                       get_format_term,
                       get_visit)
from product_labels.suffix_info import (ACCEPTED_SUFFIXES,
                                        ACCEPTED_LETTER_CODES,
                                        INSTRUMENT_FROM_LETTER_CODE)
from queue_manager.task_queue_db import remove_all_task_queue_for_a_prog_id

def ymd_tuple_to_mjd(ymd):
    """Return Modified Julian Date.

    Input:
        ymd    a tuple of year, month, and day.
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
        proposal_id    a proposal id.
        instrument     a instrument name.
        start_date     observation start datetime.
        end_date       observation end datetime.
        logger         pdslogger to use; None for default EasyLogger.
        max_retries    number of retries when there is a connection to mast.
    """
    logger = logger or pdslogger.EasyLogger()
    start_date = ymd_tuple_to_mjd(start_date)
    end_date = ymd_tuple_to_mjd(end_date)
    retry = 0
    logger.info('Query MAST: run query_mast_slice')

    query_params = {
        'dataRights': 'PUBLIC',
        'obs_collection': ['HST'],
        # 't_obs_release': (start_date, end_date),
        'mtFlag': True
    }

    if proposal_id is not None:
        formatted_proposal_id = get_formatted_proposal_id(proposal_id)
        query_params['proposal_id'] = formatted_proposal_id
    if instrument is not None:
        query_params['instrument_name'] = str(instrument)
    if start_date is not None and end_date is not None:
        query_params['t_obs_release'] = (start_date, end_date)

    for retry in range(max_retries):
        try:
            if testing and max_retries > 1:
                raise ConnectionError
            table = Observations.query_criteria(**query_params)
            return table
        except ConnectionError as e:
            retry = retry + 1
            logger.info(f'retry #{retry}: {e}')
            time.sleep(1)

    # Before raising the error, remove the task queue of the proposal id from database.
    # TODO: Maybe just update the task status from running to waiting, so we can restart
    # from current failed task when restarting the pipeline.
    if proposal_id is not None:
        remove_all_task_queue_for_a_prog_id(formatted_proposal_id)

    logger.exception(RuntimeError)
    raise RuntimeError('Query mast timed out. Number of retries: ' + str(max_retries))

def filter_table(row_predicate, table):
    """Return a copy of the filtered table object based on the return of row_predicate.

    Input:
        row_predicate    a function with the condition used to filter the table.
        table            target table to be filtered.
    """
    to_delete = [n for (n, row) in enumerate(table) if not row_predicate(row)]
    copy = table.copy()
    copy.remove_rows(to_delete)
    return copy

def is_accepted_instrument_letter_code(row):
    """Check if a product row has accepted letter code in the first letter of the
    product filename.

    Input:
        row    an observation table row.
    """
    return row['obs_id'][0].lower() in ACCEPTED_LETTER_CODES

def is_accepted_instrument_suffix(row):
    """Check if a product row has accepted suffex in the productSubGroupDescription field
    of the table.

    Input:
        row    an observation table row.
    """
    suffix = get_suffix(row)
    instrument_id = get_instrument_id_from_table_row(row)
    # For files like n4wl03fxq_raw.jpg with '--' will raise an error
    # return is_accepted(suffix, instrument_id)
    return suffix in ACCEPTED_SUFFIXES[instrument_id]

def is_trl_suffix(row):
    """Check if a product row has trl in the productSubGroupDescription field
    of the table.

    Input:
        row    an observation table row.
    """
    return get_suffix(row) == 'trl'

def is_targeted_visit(row, visit):
    """Check if a product row is related to a given visit.

    Input:
        row      an observation table row.
        visit    two character visit.
    """
    filename = row['productFilename']
    format_term = get_format_term(filename)
    return get_visit(format_term) == str(visit)

def get_instrument_id_from_table_row(row):
    """Return the instrument id for a given product row.

    Input:
        row    an observation table row.
    """
    return INSTRUMENT_FROM_LETTER_CODE[row['obs_id'][0].lower()]

def get_suffix(row):
    """Return the product file suffix for a given product row.

    Input:
        row    an observation table row.
    """
    return str(row['productSubGroupDescription']).lower()

def get_filtered_products(table, visit=None):
    """Return product rows of an observation table with accepted instrument letter code
    and suffxes. If visit is specified, only return the product rows of the targeted
    visit.

    Input:
        table    an observation table from mast query.
        visit    two character visit.
    """
    result = Observations.get_product_list(table)
    result = filter_table(is_accepted_instrument_letter_code, result)
    result = filter_table(is_accepted_instrument_suffix, result)
    if visit is not None:
        # to_delete = [n for (n, row) in enumerate(result) if not is_targeted_visit(row, visit)]
        to_delete = []
        for (n, row) in enumerate(result):
            if not is_targeted_visit(row, visit):
                to_delete.append(n)
        copy = result.copy()
        copy.remove_rows(to_delete)
        result = copy
    return result

def get_trl_products(table):
    """Return product rows of an observation table with trl suffix.

    Input:
        table    an observation table from mast query.
    """
    result = Observations.get_product_list(table)
    result = filter_table(is_accepted_instrument_letter_code, result)
    result = filter_table(is_trl_suffix, result)
    return result

def download_files(table, dir, logger=None, testing=False):
    """Download files from mast for a given product table and proposal id.

    Input:
        table          an observation table from mast query.
        proposal_id    a proposal id.
        dir            the directory we want to store the downloaded files.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    # When there is 0 product row from query result, we don't create the directory
    if len(table) == 0:
        logger.warn('Empty result from mast query')
        return
    os.makedirs(dir, exist_ok=True)

    if len(table) > 0:
        logger.info('Download files to ' + dir)
        if not testing: # pragma: no cover, no need to download files during the test
            Observations.download_products(table, download_dir=dir)
