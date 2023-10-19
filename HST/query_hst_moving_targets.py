##########################################################################################
# query_hst_moving_targets.py
#
# query_hst_moving_targets is the main function called in query_hst_moving_targets
# pipeline task script.
# It'll get a list of proposal ids from the MAST query with the given query constraints.
##########################################################################################

import pdslogger

from hst_helper import (START_DATE,
                         END_DATE,
                         RETRY)
from hst_helper.query_utils import query_mast_slice

def query_hst_moving_targets(proposal_ids=[],
                             instruments=[],
                             start_date=START_DATE,
                             end_date=END_DATE,
                             logger=None,
                             max_retries=RETRY):
    """Return a list of proposal ids with moving targets.

    Input:
        proposal_ids    a list of proposal ids.
        instruments     a list of instruments.
        start_date      observation start datetime.
        end_date        observation end datetime.
        logger          pdslogger to use; None for default EasyLogger.
        max_retries     number of retries when there is a connection to MAST.

    Returns:    a list of proposal ids with moving targets.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info('Run query_hst_moving_targets')

    table = []
    if len(proposal_ids) == 0 and len(instruments) == 0:
        # Query MAST for data observed between start_date & end_date
        res = query_mast_slice(start_date=start_date,
                               end_date=end_date,
                               logger=logger,
                               max_retries=max_retries)
        if len(res) != 0:
            table.append(res)
    elif len(instruments) == 0:
        # Query MAST for each proposal id
        for id in proposal_ids:
            res = query_mast_slice(proposal_id=id,
                                   start_date=start_date,
                                   end_date=end_date,
                                   logger=logger,
                                   max_retries=max_retries)
            if len(res) != 0:
                table.append(res)
    elif len(proposal_ids) == 0:
        # Query MAST for each instrument
        for inst in instruments:
            res = query_mast_slice(instrument=inst,
                                   start_date=start_date,
                                   end_date=end_date,
                                   logger=logger,
                                   max_retries=max_retries)
            if len(res) != 0:
                table.append(res)
    else:
        # Query MAST for all the combinations of proposal ids & instruments
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

    # Get all the unique proposal ids from tables obtained from MAST query
    p_id_li = []
    for t in table:
        for row in t:
            p_id = row['proposal_id'].zfill(5)
            if p_id not in p_id_li:
                p_id_li.append(p_id)

    return p_id_li
