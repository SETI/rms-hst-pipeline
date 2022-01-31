import unittest

from pdart.pds4.hst_filename import HstFilename

_TEST_FILENAME = "/some/random/folder/j6gp01mmq_trl.fits"


class TestHstFilename(unittest.TestCase):
    def test_init(self) -> None:
        # test length
        with self.assertRaises(Exception):
            HstFilename("123456")

        # test instrument name
        with self.assertRaises(Exception):
            HstFilename("e123456")
        HstFilename("I123456")  # case-less

    def test_str(self) -> None:
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual(_TEST_FILENAME, hst.__str__())

    def test_repr(self) -> None:
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual("HstFilename('" + _TEST_FILENAME + "')", repr(hst))

    def test_instrument_name(self) -> None:
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual("acs", hst.instrument_name())

    def test_rootname(self) -> None:
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual("j6gp01mmq", hst.rootname())

    def test_hst_internal_proposal_id(self) -> None:
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual("6gp", hst.hst_internal_proposal_id())

    def test_visit(self) -> None:
        hst = HstFilename(_TEST_FILENAME)
        self.assertEqual("01", hst.visit())
