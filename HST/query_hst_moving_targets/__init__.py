##########################################################################################
# query_hst_moving_targets/__init__.py
##########################################################################################
import time
import os

import pdslogger

from astroquery.mast import Observations
from requests.exceptions import ConnectionError

from query_mast.utils import (filter_table,
                    is_accepted_instrument_letter_code,
                    is_accepted_instrument_suffix,
                    ymd_tuple_to_mjd)

TWD = os.environ["TMP_WORKING_DIR"]
DEFAULT_DIR = TWD + "/files_from_mast"
START_DATE = (1900, 1, 1)
END_DATE = (2025, 1, 1)

def query_hst_moving_targets(proposal_ids=[],
                             instruments=[],
                             start_date=START_DATE,
                             end_date=END_DATE,
                             logger=None,
                             max_retries=1):
    """Task: query-hst-moving-targets
    Return a list of proposal ids with moving targets.
    Input:
        proposal_ids:   a list of proposal ids.
        instruments:    a list of instruments.
        start_date:     observation start datetime.
        end_date:       observation end datetime.
        logger:         pdslogger to use; None for default EasyLogger.
        max_retries:    number of retries when there is a connection to mast.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info("Run query_hst_moving_targets")

    table = []
    if len(proposal_ids) == 0 and len(instruments) == 0:
        # Query Mast for data observed between start_date & end_date
        res = query_mast_slice(start_date=start_date,
                               end_date=end_date,
                               logger=logger,
                               max_retries=max_retries)
        if len(res) != 0:
            table.append(res)
    elif len(instruments) == 0:
        # Query Mast for each proposal id
        for id in proposal_ids:
            res = query_mast_slice(proposal_id=id,
                                   start_date=start_date,
                                   end_date=end_date,
                                   logger=logger,
                                   max_retries=max_retries)
            if len(res) != 0:
                table.append(res)
    elif len(proposal_ids) == 0:
        # Query Mast for each instrument
        for inst in instruments:
            res = query_mast_slice(instrument=inst,
                                   start_date=start_date,
                                   end_date=end_date,
                                   logger=logger,
                                   max_retries=max_retries)
            if len(res) != 0:
                table.append(res)
    else:
        # Query Mast for all the combinations of proposal ids & instruments
        for id in proposal_ids:
            for inst in instruments:
                res = query_mast_slice(proposal_id=id,
                                       instrument=inst,
                                       start_date=start_date,
                                       end_date=end_date,
                                       logger=logger,
                                       max_retries=max_retries)
                if len(res) != 0:
                    table.append(res)

    # Get all the unique proposal ids from tables obtained from Mast query
    p_id_li = []
    for t in table:
        for row in t:
            p_id = row["proposal_id"].zfill(5)
            if p_id not in p_id_li:
                p_id_li.append(p_id)

    return p_id_li

def query_mast_slice(proposal_id=None,
                     instrument=None,
                     start_date=START_DATE,
                     end_date=END_DATE,
                     logger=None,
                     max_retries=1,
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
    logger.info("Run query_mast_slice")

    query_params = {
        "dataRights": "PUBLIC",
        "obs_collection": ["HST"],
        "t_obs_release": (start_date, end_date),
        "mtFlag": True
    }

    if proposal_id is not None:
        query_params["proposal_id"] = str(proposal_id)
    if instrument is not None:
        query_params["instrument_name"] = str(instrument)

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