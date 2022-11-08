from query_mast import (get_products_from_mast,
                        START_DATE,
                        END_DATE,
                        DEFAULT_DIR)
import pytest

class TestQueryMast:
    @pytest.mark.parametrize(
        'proposal_id,error_code',
        [
            (['7885'], None),
            (['wrong_proposal_id'], 1),
            ('7885', 2),
        ]
    )
    def test_get_products_from_mast(self, proposal_id, error_code):
        if error_code is None:
            get_products_from_mast(proposal_id=proposal_id, testing=True)
        elif error_code == 1:
            with pytest.raises(ValueError):
                get_products_from_mast(proposal_id=proposal_id, testing=True)
        elif error_code == 2:
            with pytest.raises(RuntimeError):
                get_products_from_mast(proposal_id=proposal_id, 
                                       max_retries=2, 
                                       testing=True)    

    

    
