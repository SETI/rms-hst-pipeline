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
from finalize_context import label_hst_context_directory
from finalize_document import label_hst_document_directory
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

    # Test schema label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_schema_directory(self, p_id):
        label_path = label_hst_schema_directory(p_id, self.data_dict, None, True)

        if os.path.isfile(label_path):
            calculated_contents = golden_file_contents(label_path)
        assert_golden_file_equal("test_schema_col_label.golden.xml", calculated_contents)

    # Test context label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_context_directory_ctxt_lbl(self, p_id):
        data_dict = {
            'collection_name': 'context',
            'csv_filename': 'collection_context.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        label_path, _ = label_hst_context_directory(p_id, data_dict, None, True)

        if os.path.isfile(label_path):
            calculated_contents = golden_file_contents(label_path)
        assert_golden_file_equal("test_context_col_label.golden.xml", calculated_contents)

    # Test investigation label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_context_directory_inv_lbl(self, p_id):
        data_dict = {
            'collection_name': 'context',
            'csv_filename': 'collection_context.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        _, label_path = label_hst_context_directory(p_id, data_dict, None, True)

        if os.path.isfile(label_path):
            calculated_contents = golden_file_contents(label_path)
        assert_golden_file_equal("test_investigation_col_label.golden.xml",
                                 calculated_contents)

    # Test document label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_document_directory_inv_lbl(self, p_id):
        data_dict = {
            'collection_name': 'document',
            'csv_filename': 'collection_document.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        label_path = label_hst_document_directory(p_id, data_dict, None, True)

        if os.path.isfile(label_path):
            calculated_contents = golden_file_contents(label_path)
        assert_golden_file_equal("test_document_col_label.golden.xml",
                                 calculated_contents)
