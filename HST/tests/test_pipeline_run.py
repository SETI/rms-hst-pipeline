##########################################################################################
# tests/test_pipeline_run.py
#
# Pipeline test run on pre-downloaded MAST files. If the pre-downloaded files don't exist,
# downlaoding will be kicked start during the test
##########################################################################################

import pdslogger
import pytest
import shutil

from finalize_hst_bundle import finalize_hst_bundle
from get_program_info import get_program_info
from hst_helper import (START_DATE,
                        END_DATE,
                        RETRY)
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path)
from prepare_browse_products import prepare_browse_products
from product_labels import label_hst_fits_directories
from query_hst_moving_targets import query_hst_moving_targets
from query_hst_products import query_hst_products
from queue_manager import run_pipeline
from retrieve_hst_visit import retrieve_hst_visit

# from update_hst_visit import update_hst_visit

class TestPipeline:
    def setup_method(self):
        # Remove the corresponding directories in HST_PIPELINE & HST_BUNDLES if they
        # exist
        self.pipeline_dir = get_program_dir_path('7885', None, 'pipeline')
        shutil.rmtree(self.pipeline_dir, ignore_errors=True)
        self.bundle_dir = get_program_dir_path('7885', None, 'bundles')
        shutil.rmtree(self.bundle_dir, ignore_errors=True)

    # Test run pipeline by calling run_pipeline
    @pytest.mark.parametrize('p_ids', [(['7885'])])
    def test_pipeline_run(self, p_ids):
        logger = pdslogger.PdsLogger('pds.hst.run-pipeline')
        try:
            run_pipeline(p_ids, logger)
        except Exception as e:
            assert False, f'Pipeline on {p_ids} has error: {e}'

    # Test run pipeline by calling each function
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_pipeline_run_all_functions(self, p_id):
        shutil.rmtree(self.pipeline_dir, ignore_errors=True)
        shutil.rmtree(self.bundle_dir, ignore_errors=True)
        logger = pdslogger.PdsLogger('pds.hst.run-pipeline-all-functions')
        try:
            formatted_proposal_id = get_formatted_proposal_id(p_id)
            query_hst_moving_targets(proposal_ids=[p_id],
                                  instruments=[],
                                  start_date=START_DATE,
                                  end_date=END_DATE,
                                  logger=logger,
                                  max_retries=RETRY)
            _, all_visits = query_hst_products(p_id, logger)
            get_program_info(formatted_proposal_id, None, logger)

            for visit in all_visits:
                retrieve_hst_visit(formatted_proposal_id, visit, logger)
                path = get_program_dir_path(formatted_proposal_id, visit, 'staging')
                label_hst_fits_directories(path)
                prepare_browse_products(formatted_proposal_id, visit, logger)

            finalize_hst_bundle(formatted_proposal_id, logger)

        except Exception as e:
            assert False, f'Pipeline on {p_id} has error: {e}'
