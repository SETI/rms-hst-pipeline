##########################################################################################
# tests/test_label_creations_with_fits_file.py
#
# Tests related to label creations using fits files.
##########################################################################################

import os
import pytest
import shutil

from product_labels import label_hst_fits_directories
from .utils import (assert_golden_file_equal,
                    golden_filepath,
                    golden_file_contents,
                    TEST_COPIES_DIR)


class TestLabelCreations:
    # def setup_method(self):


    def teardown_method(self):
        # Remove the created xml
        created_lbl = [
            f'{TEST_COPIES_DIR}/hst_07885/visit_01/mastDownload/HST/n4wl01xdq/n4wl01xdq_spt.xml',
            f'{TEST_COPIES_DIR}/hst_07885/visit_01/mastDownload/HST/n4wl01xdq/n4wl01xdq_raw.xml',
            f'{TEST_COPIES_DIR}/hst_07885/visit_01/mastDownload/HST/n4wl01xdq/n4wl01xdq_trl.xml',
        ]
        created_lbl_paths = [golden_filepath(lbl_path) for lbl_path in created_lbl]
        for lbl in created_lbl_paths:
            try:
                os.remove(lbl)
            except FileNotFoundError:
                pass

    # Test label creation using fits file
    @pytest.mark.parametrize(
        'paths,expected',
        [
            ([f'{TEST_COPIES_DIR}/hst_07885'],
             {
                f'{TEST_COPIES_DIR}/hst_07885/visit_01/mastDownload/HST/n4wl01xdq/n4wl01xdq_spt.xml':
                f'{TEST_COPIES_DIR}/test_n4wl01xdq_spt.golden.xml',
                f'{TEST_COPIES_DIR}/hst_07885/visit_01/mastDownload/HST/n4wl01xdq/n4wl01xdq_raw.xml':
                f'{TEST_COPIES_DIR}/test_n4wl01xdq_raw.golden.xml',
                f'{TEST_COPIES_DIR}/hst_07885/visit_01/mastDownload/HST/n4wl01xdq/n4wl01xdq_trl.xml':
                f'{TEST_COPIES_DIR}/test_n4wl01xdq_trl.golden.xml',
             }),
        ]
    )
    def test_label_hst_prod_directory(self, paths, expected):
        paths = [golden_filepath(path) for path in paths]

        expected_li = []
        for lbl_path in expected.keys():
            created_lbl_path = golden_filepath(lbl_path)
            expected_li.append((created_lbl_path, expected[lbl_path]))

        label_hst_fits_directories(directories=paths)

        for created_lbl_path, golden_copy_path in expected_li:
            if os.path.isfile(created_lbl_path):
                calculated_contents = golden_file_contents(created_lbl_path)

            assert_golden_file_equal(golden_copy_path,
                                     calculated_contents)
