from query_hst_moving_targets import query_hst_moving_targets
import pytest


class TestQueryHSTMovingTargets:
    @pytest.mark.parametrize(
        'p_ids,inst,expected',
        [
            ([], [], 781),
            ([], ['NICMOS/NIC1'], 24),
            (['7885'], [], 1),
            (['15505'], ['WFC3/UVIS'], 1),
        ],
    )
    def test_query_hst_moving_targets(self, p_ids, inst, expected):
        id_li = query_hst_moving_targets(proposal_ids=p_ids, instruments=inst)
        print(len(id_li))
        assert len(id_li) == expected
