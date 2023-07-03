##########################################################################################
# tests/test_get_program_info.py
#
# Tests related to get_prgroam_info task
##########################################################################################

import pytest
import shutil
import tempfile

from get_program_info import get_program_info

class TestGetProgramInfo:
    def setup_method(self):
        self.download_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.download_dir)

    @pytest.mark.parametrize(
        'p_id,expected',
        [
            ('7885', {"07885.pro"}),
            ('9748', {"09748.apt", "09748.pro"}),
        ],
    )
    def test_get_program_info(self, p_id, expected):
        res = get_program_info(p_id, self.download_dir)
        print(res)
        assert expected == res
