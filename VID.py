import re

class VID:
    def __init__(self, str):
	vs = str.split('.')

	# Check requirements
	assert len(str) <= 255
	assert len(vs) == 2
	for v in vs:
	    assert re.match('\\A(0|[1-9][0-9]*)\\Z', v)

	self.VID = str
	self.major = int(vs[0])
	self.minor = int(vs[1])

    def __eq__(self, other):
	return self.VID == other.VID

    def __ne__(self, other):
	return self.VID != other.VID

    def __str__(self):
	return self.VID

    def __repr__(self):
	return 'VID(%s)' % repr(self.VID)

############################################################

import unittest

class TestVID(unittest.TestCase):
    def testInit(self):
	# sanity-check
	with self.assertRaises(Exception):
	    VID(null)

	with self.assertRaises(Exception):
	    VID('foo')

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
        self.assertEqual(3, v.major)
        self.assertEqual(14159265, v.minor)

    def testEq(self):
        self.assertTrue(VID('2.3') == VID('2.3'))
        self.assertTrue(VID('2.3') != VID('2.4'))
        self.assertFalse(VID('2.3') == VID('3.2'))
        self.assertFalse(VID('2.3') != VID('2.3'))

    def testStr(self):
        self.assertEquals('2.3', str(VID('2.3')))

    def testRepr(self):
        self.assertEquals("VID('2.3')", repr(VID('2.3')))

if __name__ == '__main__':
    unittest.main()
