import unittest

from pdart.astroquery.Astroquery import *


class TestAstroquery(unittest.TestCase):
    def test_init(self):
        start_date = (1900, 1, 1)
        end_date = (2018, 3, 26)
        slice = MastSlice(start_date, end_date)
        self.assertTrue(len(slice.observations_table) >= 20000)
