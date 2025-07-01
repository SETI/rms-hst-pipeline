##########################################################################################
# tests/test_query_hst_moving_targets.py
#
# Tests related to query_hst_moving_targets task
##########################################################################################

import pytest
from hst_helper import (START_DATE,
                        END_DATE)
from query_hst_moving_targets import query_hst_moving_targets

class TestQueryHSTMovingTargets:
    @pytest.mark.parametrize(
        'p_ids,inst,start_date,end_date,expected',
        [
            ([], ['NICMOS/NIC1'], START_DATE, END_DATE, 24),
            ([], ['NICMOS/NIC1'], (1900, 1, 1), (1900, 1, 2), 0),
            (['7885'], [], START_DATE, END_DATE, 1),
            (['06774'], [], (1900, 1, 1), (1900, 1, 2), 0),
            (['15505'], ['WFC3/UVIS'], START_DATE, END_DATE, 1),
            (['15505'], ['WFC3/UVIS'], (1900, 1, 1), (1900, 1, 2), 0),
            ([], [], (1900, 1, 1), (1900, 1, 2), 0),
            ([], [], (1998, 1, 1), (1999, 1, 1), 42),
        ],
    )
    def test_query_hst_moving_targets(self, p_ids, inst, start_date, end_date, expected):
        id_li = query_hst_moving_targets(proposal_ids=p_ids,
                                         instruments=inst,
                                         start_date=start_date,
                                         end_date=end_date)
        print(len(id_li))
        assert len(id_li) == expected
