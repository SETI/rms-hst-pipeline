import shutil
import tempfile
import unittest

from pdart.pipeline.MarkerFile import *


class Test_MarkerFile(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp(None, "test_markerfile_")
        print("****", self.tempdir)
        self.mf = BasicMarkerFile(self.tempdir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def test_get_marker(self) -> None:
        mf: MarkerFile = self.mf
        # Test that get on nothing returns None
        self.assertIsNone(mf.get_marker())

    def test_clear_marker(self) -> None:
        mf: MarkerFile = self.mf
        # Test that clear on nothing is harmless
        mf.clear_marker()

        # Test that set then clear then get returns nothing
        mf.set_marker(MarkerInfo("foo", "bar", "baz"))
        mf.clear_marker()
        m = mf.get_marker()
        self.assertIsNone(m)

    def test_set_marker(self) -> None:
        mf: MarkerFile = self.mf

        # Test that set followed by get returns set's argument
        mf.set_marker(MarkerInfo("foo", "bar", "baz"))
        m = mf.get_marker()
        self.assertIsNotNone(m)
        # Superfluous test to let mypy know m != None
        if m is not None:
            self.assertEqual("FOO", m.phase)
            self.assertEqual("BAR", m.state)
            self.assertEqual("baz", m.text)

        # Test that a second set yields new info
        mf.set_marker(MarkerInfo("one", "two", "Three!"))
        m = mf.get_marker()
        self.assertIsNotNone(m)
        # Superfluous test to let mypy know m != None
        if m is not None:
            self.assertEqual("ONE", m.phase)
            self.assertEqual("TWO", m.state)
            self.assertEqual("Three!", m.text)
