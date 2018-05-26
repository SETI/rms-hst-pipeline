import unittest

from pdart.astroquery.Astroquery import *


class TestAstroquery(unittest.TestCase):
    def setUp(self):
        start_date = (1900, 1, 1)
        end_date = (2018, 3, 26)
        self.slice = MastSlice(start_date, end_date)

    def test_init(self):
        self.assertTrue(len(self.slice.observations_table) >= 20000)

    def test_get_proposal_ids(self):
        self.assertTrue(len(self.slice.get_proposal_ids()) >= 500)

    def test_get_products(self):
        proposal_ids = self.slice.get_proposal_ids()
        for proposal_id in proposal_ids:
            # smoke test only: runs it and quits
            products_table = self.slice.get_products(proposal_id)
            return
