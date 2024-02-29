import unittest

from astropy.table import Table

from pdart.astroquery.astroquery import MastSlice


@unittest.skip("really slow: investigate")
class TestAstroquery(unittest.TestCase):
    slice: MastSlice

    @classmethod
    def setUpClass(cls) -> None:
        # It's relatively expensive to make the MAST call, and it
        # never changes, so we're going to reuse the same one for all
        # tests.
        start_date = (1900, 1, 1)
        end_date = (2018, 3, 26)
        cls.slice = MastSlice(start_date, end_date)

    def test_observation_column_names(self) -> None:
        # This test is just to let us know when the column names for
        # observations returned from MAST changes.  If it fails, no
        # big deal: it's just a heads-up for a human to note the
        # change and verify that it's not significant for us.
        expected = [
            "calib_level",
            "dataRights",
            "dataURL",
            "dataproduct_type",
            "em_max",
            "em_min",
            "filters",
            "instrument_name",
            "intentType",
            "jpegURL",
            "mtFlag",
            "objID",
            "obs_collection",
            "obs_id",
            "obs_title",
            "obsid",
            "project",
            "proposal_id",
            "proposal_pi",
            "proposal_type",
            "provenance_name",
            "s_dec",
            "s_ra",
            "s_region",
            "sequence_number",
            "srcDen",
            "t_exptime",
            "t_max",
            "t_min",
            "t_obs_release",
            "target_classification",
            "target_name",
            "wavelength_region",
        ]
        actual = sorted(TestAstroquery.slice.observations_table.colnames)
        self.assertEqual(expected, actual)

    def test_product_column_names(self) -> None:
        # This test is just to let us know when the column names for
        # products returned from MAST changes.  If it fails, no big
        # deal: it's just a heads-up for a human to note the change
        # and verify that it's not significant for us.
        def get_a_products_table() -> Table:
            slice = TestAstroquery.slice
            proposal_id = slice.get_proposal_ids()[0]
            return slice.get_products(proposal_id)

        expected = [
            "dataRights",
            "dataURI",
            "dataproduct_type",
            "description",
            "obsID",
            "obs_collection",
            "obs_id",
            "parent_obsid",
            "productDocumentationURL",
            "productFilename",
            "productGroupDescription",
            "productSubGroupDescription",
            "productType",
            "project",
            "proposal_id",
            "prvversion",
            "size",
            "type",
        ]

        actual = sorted(get_a_products_table().colnames)
        self.assertEqual(expected, actual)

    def test_init(self) -> None:
        self.assertTrue(len(TestAstroquery.slice.observations_table) >= 20000)

    def test_get_proposal_ids(self) -> None:
        self.assertTrue(len(TestAstroquery.slice.get_proposal_ids()) >= 500)

    def test_get_products(self) -> None:
        proposal_ids = TestAstroquery.slice.get_proposal_ids()
        for proposal_id in proposal_ids:
            # smoke test only: runs it and quits
            products_table = TestAstroquery.slice.get_products(proposal_id)
            return
