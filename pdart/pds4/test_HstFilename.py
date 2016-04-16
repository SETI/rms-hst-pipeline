import unittest

from pdart.pds4.HstFilename import *

_TEST_FILENAME = '/some/random/folder/j6gp01mmq_trl.fits'


class TestHstFilename(unittest.TestCase):
    def test_init(self):
        # test bad type
        with self.assertRaises(Exception):
            HstFilename(None)
        with self.assertRaises(Exception):
            HstFilename(1)

        # test length
        with self.assertRaises(Exception):
            HstFilename('123456')

        # test instrument name
        with self.assertRaises(Exception):
            HstFilename('x123456')
        HstFilename('I123456')  # case-less

    def test_str(self):
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual(_TEST_FILENAME, hst.__str__())

    def test_repr(self):
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual('HstFilename(\'' + _TEST_FILENAME + '\')', repr(hst))

    def test_instrument_name(self):
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual('acs', hst.instrument_name())

    def test_hst_internal_proposal_id(self):
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual('6gp', hst.hst_internal_proposal_id())

    def test_visit(self):
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual('01', hst.visit())
