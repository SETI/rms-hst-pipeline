##########################################################################################
# tests/test_label_creations.py
#
# Tests related to label creations in the final bundle directory.
##########################################################################################

import os
import pytest
import shutil

from .utils import (assert_golden_file_equal,
                    golden_file_contents,
                    LBL_DATA_DICT)
from finalize_schema import label_hst_schema_directory
from hst_helper.fs_utils import get_program_dir_path

class TestLabelCreations:
    def setup_method(self):
        # data dictionary used to create the label
        self.data_dict = LBL_DATA_DICT

        # Make the temporary directories for testing
        self.testing_dir = [get_program_dir_path('7885', None, 'bundles', True)]
        for temp_dir in self.testing_dir:
            os.mkdir(temp_dir)

    def teardown_method(self):
        # Remove the testing directories
        for testing_dir in self.testing_dir:
            shutil.rmtree(testing_dir)

    @pytest.mark.parametrize(
        'p_id',
        [
            ('7885'),
        ],
    )
    def test_label_hst_schema_directory(self, p_id):
        label_path = label_hst_schema_directory(p_id, self.data_dict, None, True)

        if os.path.isfile(label_path):
            calculated_contents = golden_file_contents(label_path)
        assert_golden_file_equal("test_schema_label.golden.xml", calculated_contents)
