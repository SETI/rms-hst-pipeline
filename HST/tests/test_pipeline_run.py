##########################################################################################
# tests/test_pipeline_run.py
#
# Pipeline test run on pre-downloaded MAST files. If the pre-downloaded files don't exist,
# downlaoding will be kicked start during the test.
# Note: each pipeline test will take about 3 minutes.
##########################################################################################

import pdslogger
import pytest

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
from .utils import remove_dirs

class TestPipeline:
    def setup_method(self):
        # Remove the corresponding directories in HST_PIPELINE & HST_BUNDLES if they
        # exist
        p_ids = ['16167']
        # p_ids = ['7885', '16167']
        self.pipeline_dirs = [get_program_dir_path(id, None, 'pipeline') for id in p_ids]
        self.bundle_dirs = [get_program_dir_path(id, None, 'bundles') for id in p_ids]

        remove_dirs(self.pipeline_dirs)
        remove_dirs(self.bundle_dirs)

    # Test run pipeline by calling run_pipeline
    @pytest.mark.parametrize('p_ids', [(['16167'])])
    def test_pipeline_run(self, p_ids):
        logger = pdslogger.PdsLogger('pds.hst.run-pipeline')
        try:
            run_pipeline(p_ids, logger)
        except Exception as e:
            assert False, f'Pipeline on {p_ids} has error: {e}'

    # Test run pipeline by calling each function
    @pytest.mark.parametrize('p_id', [('16167')])
    def test_pipeline_run_by_calling_each_function(self, p_id):
        remove_dirs(self.pipeline_dirs)
        remove_dirs(self.bundle_dirs)
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
