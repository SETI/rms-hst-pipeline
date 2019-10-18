import unittest
from hypothesis import assume, given
import hypothesis.strategies as st

from pdart.pds4.VID import VID

@st.composite
def vid_strings(draw, max_value=9):
    first = draw(st.integers(min_value=1, max_value=max_value))
    rest = draw(st.lists(st.integers(min_value=0, max_value=max_value),
                         min_size=0, max_size=3))
    if rest:
        res = '%d.%s' % (first, '.'.join([str(n) for n in rest]))
    else:
        res = str(first)
    assume(len(res) <= 255)
    return res

def pdart_vid_strings(max_value=9):
    '''
    A Hypothesis strategy to generate VID strings with exactly two
    components
    '''
    return st.builds(lambda maj, min: '%d.%d' % (maj, min),
                     st.integers(min_value=1, max_value=max_value),
                     st.integers(min_value=0, max_value=max_value))

def pdart_vids(max_value=9):
    '''
    A Hypothesis strategy to generate VIDs with exactly two
    components
    '''
    return st.builds(VID, pdart_vid_strings(max_value=max_value))

class TestVID(unittest.TestCase):
    def test_init(self):
        # type: () -> None
        # sanity-check
        with self.assertRaises(Exception):
            VID(None)

        with self.assertRaises(Exception):
            VID('foo')

        with self.assertRaises(Exception):
            VID('0.0')
        with self.assertRaises(Exception):
            VID('0.0.0')
        with self.assertRaises(Exception):
            VID('5.')
        with self.assertRaises(Exception):
            VID('.5')
        with self.assertRaises(Exception):
            VID('0.01')

        # test fields
        v = VID('3.14159265')
        self.assertEqual(3, v._major)
        self.assertEqual(14159265, v._minor)

    def test_next_major_vid(self):
        # type: () -> None
        self.assertEqual(VID('3.0'), VID('2.9').next_major_vid())

    @given(pdart_vids())
    def test_next_major_vid_property(self, vid):
        next = vid.next_major_vid()
        # The major version should increment
        self.assertEquals(next.major(), vid.major() + 1)
        # and the minor should be zero
        self.assertEquals(0, next.minor())

    def test_next_minor_vid(self):
        # type: () -> None
        self.assertEquals(VID('2.1'), VID('2.0').next_minor_vid())
        self.assertEquals(VID('2.10'), VID('2.9').next_minor_vid())

    @given(pdart_vids())
    def test_next_minor_vid_property(self, vid):
        next = vid.next_minor_vid()
        # The major version should not change
        self.assertEquals(next.major(), vid.major())
        # and the minor should increment
        self.assertEquals(next.minor(), vid.minor() + 1)

    def test_cmp(self):
        # type: () -> None
        self.assertTrue(VID('2.3') == VID('2.3'))
        self.assertTrue(VID('2.3') != VID('2.4'))
        self.assertTrue(VID('2.3') < VID('3.2'))
        self.assertTrue(VID('2.3') > VID('2.2'))

    @given(pdart_vids(), pdart_vids())
    def test_cmp_property(self, lhs, rhs):
        def cmp2(lhs, rhs):
            '''Compare, but force it to be between -1 and 1.'''
            n = cmp(lhs, rhs)
            if n < 0:
                return -1
            elif n == 0:
                return 0
            else:
                return 1

        # Comparing two VIDs should be the same as comparing their
        # version numbers.
        self.assertEquals(cmp2(lhs,rhs), cmp2([lhs.major(), lhs.minor()],
                                              [rhs.major(), rhs.minor()]))

    def test_str(self):
        # type: () -> None
        self.assertEquals('2.3', str(VID('2.3')))

    @given(pdart_vid_strings())
    def test_str_roundtrip_property(self, vid_str):
        '''
        Creating a VID from a string and turning it back into a string
        should result in the same string.
        '''
        self.assertEquals(vid_str, str(VID(vid_str)))

    def test_repr(self):
        # type: () -> None
        self.assertEquals("VID('2.3')", repr(VID('2.3')))
