import os.path
import re
import unittest


class HstFilename(object):
    s = '/some/random/folder/j6gp01mmq_trl.fits'

    def __init__(self, filename):
        self.filename = filename
        assert len(os.path.basename(filename)) > 6, \
            'Filename must be at least six characters long'
        basename = os.path.basename(filename)
        assert basename[0].lower() in 'iju', \
            ('First char of filename %s must be i, j, or u.' % str(basename))

    def __str__(self):
        return self.filename.__str__()

    def __repr__(self):
        return 'HstFilename(%r)' % self.filename

    def _basename(self):
        return os.path.basename(self.filename)

    def instrument_name(self):
        filename = self._basename()
        i = filename[0].lower()
        assert i in 'iju', ('First char of filename %s must be i, j, or u.'
                            % str(filename))
        if i == 'i':
            return 'wfc3'
        elif i == 'j':
            return 'acs'
        elif i == 'u':
            return 'wfpc2'
        else:
            raise Exception('First char of filename must be i, j, or u.')

    def hst_internal_proposal_id(self):
        return self._basename()[1:4].lower()

    def suffix(self):
        return re.match(r'\A[^_]+_([^\.]+)\..*\Z',
                        self._basename()).group(1)

    def visit(self):
        return self._basename()[4:6].lower()

############################################################


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
        s = HstFilename.s
        self.assertEqual(s, HstFilename(s).__str__())

    def test_repr(self):
        s = HstFilename.s
        self.assertEqual('HstFilename(\'' + s + '\')', repr(HstFilename(s)))

    def test_instrument_name(self):
        s = HstFilename.s
        self.assertEqual('acs', HstFilename(s).instrument_name())

    def test_hst_internal_proposal_id(self):
        s = HstFilename.s
        self.assertEqual('6gp', HstFilename(s).hst_internal_proposal_id())

    def test_visit(self):
        s = HstFilename.s
        self.assertEqual('01', HstFilename(s).visit())

if __name__ == '__main__':
    unittest.main()
