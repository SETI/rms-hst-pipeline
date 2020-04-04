import unittest
from typing import Any

import hypothesis.strategies as st
from hypothesis import assume, given

from pdart.pds4.VID import VID


@st.composite
def vid_strings(draw: Any, max_value: int = 9) -> str:
    first = draw(st.integers(min_value=1, max_value=max_value))
    rest = draw(
        st.lists(st.integers(min_value=0, max_value=max_value), min_size=0, max_size=3,)
    )
    if rest:
        tail = ".".join([str(n) for n in rest])
        res = f"{first}.{tail}"
    else:
        res = str(first)
    assume(len(res) <= 255)
    return res


def pdart_vid_strings(max_value: int = 9) -> st.SearchStrategy[str]:
    """
    A Hypothesis strategy to generate VID strings with exactly two
    components
    """
    return st.builds(
        lambda major, minor: f"{major}.{minor}",
        st.integers(min_value=1, max_value=max_value),
        st.integers(min_value=0, max_value=max_value),
    )


def pdart_vids(max_value: int = 9) -> st.SearchStrategy[VID]:
    """
    A Hypothesis strategy to generate VIDs with exactly two
    components
    """
    return st.builds(VID, pdart_vid_strings(max_value=max_value))


class TestVID(unittest.TestCase):
    def test_init(self) -> None:
        with self.assertRaises(Exception):
            VID("foo")

        with self.assertRaises(Exception):
            VID("0.0")
        with self.assertRaises(Exception):
            VID("0.0.0")
        with self.assertRaises(Exception):
            VID("5.")
        with self.assertRaises(Exception):
            VID(".5")
        with self.assertRaises(Exception):
            VID("0.01")

        # test fields
        v = VID("3.14159265")
        self.assertEqual(3, v._major)
        self.assertEqual(14159265, v._minor)

    def test_next_major_vid(self) -> None:
        self.assertEqual(VID("3.0"), VID("2.9").next_major_vid())

    @given(pdart_vids())
    def test_next_major_vid_property(self, vid: VID) -> None:
        next_vid = vid.next_major_vid()
        # The major version should increment
        self.assertEqual(next_vid.major(), vid.major() + 1)
        # and the minor should be zero
        self.assertEqual(0, next_vid.minor())

    def test_next_minor_vid(self) -> None:
        self.assertEqual(VID("2.1"), VID("2.0").next_minor_vid())
        self.assertEqual(VID("2.10"), VID("2.9").next_minor_vid())

    @given(pdart_vids())
    def test_next_minor_vid_property(self, vid: VID) -> None:
        next_vid = vid.next_minor_vid()
        # The major version should not change
        self.assertEqual(next_vid.major(), vid.major())
        # and the minor should increment
        self.assertEqual(next_vid.minor(), vid.minor() + 1)

    def test_cmp(self) -> None:
        self.assertTrue(VID("2.3") == VID("2.3"))
        self.assertTrue(VID("2.3") != VID("2.4"))
        self.assertTrue(VID("2.3") < VID("3.2"))
        self.assertTrue(VID("2.3") > VID("2.2"))

    @given(pdart_vids(), pdart_vids())
    def test_eq_property(self, lhs: VID, rhs: VID) -> None:
        # Comparing two VIDs should be the same as comparing their
        # version numbers.
        self.assertEqual(
            lhs == rhs, [lhs.major(), lhs.minor()] == [rhs.major(), rhs.minor()]
        )

    @given(pdart_vids(), pdart_vids())
    def test_lt_property(self, lhs: VID, rhs: VID) -> None:
        # Comparing two VIDs should be the same as comparing their
        # version numbers.
        self.assertEqual(
            lhs < rhs, [lhs.major(), lhs.minor()] < [rhs.major(), rhs.minor()]
        )

    def test_str(self) -> None:
        self.assertEqual("2.3", str(VID("2.3")))

    @given(pdart_vid_strings())
    def test_str_roundtrip_property(self, vid_str: str) -> None:
        """
        Creating a VID from a string and turning it back into a string
        should result in the same string.
        """
        self.assertEqual(vid_str, str(VID(vid_str)))

    def test_repr(self) -> None:
        self.assertEqual("VID('2.3')", repr(VID("2.3")))
