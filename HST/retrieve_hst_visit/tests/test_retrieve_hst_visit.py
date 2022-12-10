from retrieve_hst_visit import retrieve_hst_visit
import pytest

class TestQueryMast:
    @pytest.mark.parametrize(
        'proposal_id,visit,expected',
        [
            ('7885', '01', 105),
            ('7885', '04', 0),
        ]
    )
    def test_retrieve_hst_visit(self, proposal_id, visit, expected):
        res = retrieve_hst_visit(proposal_id, visit, testing=True)
        assert res == expected

    @pytest.mark.parametrize(
        'proposal_id,visit',
        [
            ('wrong_proposal_id', '01'),
        ]
    )
    def test_wrong_proposal_id(self, proposal_id, visit):
        with pytest.raises(ValueError):
            retrieve_hst_visit(proposal_id, visit, testing=True)
