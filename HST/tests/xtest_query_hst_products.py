from query_hst_products import query_hst_products
import pytest


class TestQueryHSTProducts:
    # TODO: Need to choose one proposal id and remove both products.txt
    # and trl_checksums.txt to cover all code
    @pytest.mark.parametrize(
        'p_id,expected',
        [
            ('9296', []),
        ],
    )
    def test_query_hst_products(self, p_id, expected):
        visit_diff, visits = query_hst_products(p_id)
        print('================')
        print(visit_diff)
        print(visits)
        # assert visit_diff == expected
