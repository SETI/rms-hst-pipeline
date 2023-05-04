##########################################################################################
# tests/test_retrieve_hst_visit.py
#
# Tests related to retrieve_hst_visit task
##########################################################################################

import os
import pytest
import shutil

from hst_helper.fs_utils import get_program_dir_path
from retrieve_hst_visit import retrieve_hst_visit


class TestQueryMast:
    def setup_method(self):
        # Make the temporary directories for testing
        self.testing_dir = [get_program_dir_path('7885', None, 'staging', True)]
        for temp_dir in self.testing_dir:
            os.mkdir(temp_dir)

    def teardown_method(self):
        # Remove the testing directories
        for testing_dir in self.testing_dir:
            shutil.rmtree(testing_dir)

    @pytest.mark.parametrize(
        'proposal_id,visit,expected',
        [
            ('7885', '01', 105),
            ('7885', '04', 0),
        ]
    )
    def test_retrieve_hst_visit(self, proposal_id, visit, expected):
        res = retrieve_hst_visit(proposal_id, visit, None, testing=True)
        assert res == expected

    @pytest.mark.parametrize(
        'proposal_id,visit',
        [
            ('wrong_proposal_id', '01'),
        ]
    )
    def test_wrong_proposal_id(self, proposal_id, visit):
        with pytest.raises(ValueError):
            retrieve_hst_visit(proposal_id, visit, None)
