import unittest
from pdart.astroquery.Utils import *


class TestUtils(unittest.TestCase):
    def test_sanity(self):
        # type: () -> None
        """
        Just a simple sanity check on converting between MJD and
        dates.
        """
        ymdhms = ymdhms_format_from_mjd(57986.72013889)
        self.assertEqual('2017-08-21T17:17:00', ymdhms)
