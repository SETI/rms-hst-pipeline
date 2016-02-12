import os.path

class HstFilename:
    s = '/some/random/folder/j6gp01mmq_trl.fits'

    def __init__(self, filename):
        self.filename = filename
        assert len(os.path.basename(filename)) > 6, 'Filename must be at least six characters long'
        basename = os.path.basename(filename)
        assert basename[0].lower() in 'iju', ('First char of filename %s must be i, j, or u.'
                                              % str(basename))

    def __str__(self):
        return self.filename.__str__()

    def __repr__(self):
        return 'HstFilename(%s)' % self.filename.__repr__()

    def __basename(self):
        return os.path.basename(self.filename)

    def instrumentName(self):
        filename = self.__basename()
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
            raise Exception, 'First char of filename must be i, j, or u.'

    def hstInternalProposalId(self):
        return self.__basename()[1:4].lower()

    def visit(self):
        return self.__basename()[4:6].lower()

############################################################

import unittest

class TestHstFilename(unittest.TestCase):
    def testInit(self):
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
        HstFilename('I123456')	# case-less

    def testStr(self):
        s = HstFilename.s
        self.assertEqual(s, HstFilename(s).__str__())

    def testRepr(self):
        s = HstFilename.s
        self.assertEqual('HstFilename(\'' + s + '\')',
                         HstFilename(s).__repr__())

    def testInstrumentName(self):
        s = HstFilename.s
        self.assertEqual('acs', HstFilename(s).instrumentName())

    def testHstInternalProposalId(self):
        s = HstFilename.s
        self.assertEqual('6gp', HstFilename(s).hstInternalProposalId())

    def testVisit(self):
        s = HstFilename.s
        self.assertEqual('01', HstFilename(s).visit())

if __name__ == '__main__':
    unittest.main()
