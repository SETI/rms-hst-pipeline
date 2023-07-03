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
from finalize_data_product import COL_DATA_LABEL_TEMPLATE
from finalize_document import label_hst_document_directory
from finalize_schema import label_hst_schema_directory
from hst_helper.fs_utils import (create_col_dir_in_bundle,
                                 get_deliverable_path,
                                 get_program_dir_path)
from hst_helper.general_utils import create_collection_label
from label_bundle import label_hst_bundle

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

    # Test schema colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_schema_directory(self, p_id):
        sch_col_lbl = label_hst_schema_directory(p_id, self.data_dict, None, True)

        if os.path.isfile(sch_col_lbl):
            calculated_contents = golden_file_contents(sch_col_lbl)
        assert_golden_file_equal("test_schema_col_label.golden.xml", calculated_contents)

    # Test context colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_context_directory_ctxt_col_lbl(self, p_id):
        data_dict = {
            'collection_name': 'context',
            'csv_filename': 'collection_context.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        ctxt_col_lbl, _ = label_hst_context_directory(p_id, data_dict, None, True)

        if os.path.isfile(ctxt_col_lbl):
            calculated_contents = golden_file_contents(ctxt_col_lbl)
        assert_golden_file_equal("test_context_col_label.golden.xml", calculated_contents)

    # Test investigation label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_context_directory_inv_lbl(self, p_id):
        data_dict = {
            'collection_name': 'context',
            'csv_filename': 'collection_context.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        _, inv_lbl = label_hst_context_directory(p_id, data_dict, None, True)

        if os.path.isfile(inv_lbl):
            calculated_contents = golden_file_contents(inv_lbl)
        assert_golden_file_equal("test_investigation_label.golden.xml",
                                 calculated_contents)

    # Test document colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_document_directory_doc_col_lbl(self, p_id):
        data_dict = {
            'collection_name': 'document',
            'csv_filename': 'collection_document.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        doc_col_lbl, _ = label_hst_document_directory(p_id, data_dict, None, True)

        if os.path.isfile(doc_col_lbl):
            calculated_contents = golden_file_contents(doc_col_lbl)
        assert_golden_file_equal("test_document_col_label.golden.xml",
                                 calculated_contents)

    # Test document label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_document_directory_doc_lbl(self, p_id):
        data_dict = {
            'collection_name': 'document',
            'csv_filename': 'collection_document.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        _, doc_lbl = label_hst_document_directory(p_id, data_dict, None, True)

        if os.path.isfile(doc_lbl):
            calculated_contents = golden_file_contents(doc_lbl)
        assert_golden_file_equal("test_document_label.golden.xml",
                                 calculated_contents)

    # Test data product colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_create_collection_label_data_prod_col_lbl(self, p_id):
        data_dict = {
            'collection_name': 'data_nicmos_cal',
            'csv_filename': 'collection_document.csv',
            'processing_level': 'Calibrated',
            'inst_id': 'nicmos',
            'collection_title': 'Calibrated NICMOS "_cal" image files from HST ' +
                                'Program 7885.',
            'instrument_name': 'Near-Infrared Camera and Multi-Object Spectrometer',
            'stop_date_time': '1998-08-05T03:02:11Z',
        }
        data_dict = {**self.data_dict, **data_dict}

        col_name = data_dict['collection_name']
        # Create data product collection directory
        create_col_dir_in_bundle(p_id, col_name, True)
        col_data_label_name = f'collection_{col_name}.xml'
        data_prod_col_lbl = create_collection_label(p_id, col_name,
                                                    data_dict, col_data_label_name,
                                                    COL_DATA_LABEL_TEMPLATE, None, True)

        if os.path.isfile(data_prod_col_lbl):
            calculated_contents = golden_file_contents(data_prod_col_lbl)
        assert_golden_file_equal("test_data_prod_col_label.golden.xml",
                                 calculated_contents)

    # Test bundle label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_bundle(self, p_id):
        data_dict = {
            'collection_name': 'bundle',
            'bundle_entry_li': [
                ('context', 'context', (1, 0)),
                ('miscellaneous_nicmos_spt', 'miscellaneous', (1, 0)),
                ('miscellaneous_nicmos_trl', 'miscellaneous', (1, 0)),
                ('miscellaneous_nicmos_jif', 'miscellaneous', (1, 0)),
                ('document', 'document', (1, 0)),
                ('miscellaneous_nicmos_jit', 'miscellaneous', (1, 0)),
                ('schema', 'schema', (1, 0)),
                ('browse_nicmos_cal', 'browse', (1, 0)),
                ('data_nicmos_ima', 'data', (1, 0)),
                ('browse_nicmos_raw', 'browse', (1, 0))
            ]
        }
        data_dict = {**self.data_dict, **data_dict}

        deliverable_path = get_deliverable_path(p_id, True)
        os.makedirs(deliverable_path, exist_ok=True)
        bundle_lbl = label_hst_bundle(p_id, data_dict, None, True)

        if os.path.isfile(bundle_lbl):
            calculated_contents = golden_file_contents(bundle_lbl)
        assert_golden_file_equal("test_bundle_label.golden.xml",
                                 calculated_contents)
