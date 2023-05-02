from query_hst_products import query_hst_products
import pytest


class TestQueryHSTProducts:
    # TODO: Need to choose one proposal id and remove both products.txt
    # and trl_checksums.txt to cover all code
    @pytest.mark.parametrize(
        'p_id,expected',
        [
            ('7885', []),
        ],
    )
    def test_query_hst_products(self, p_id, expected):
        visit_diff, _ = query_hst_products(proposal_id=p_id)
        print(visit_diff)
        assert visit_diff == expected
