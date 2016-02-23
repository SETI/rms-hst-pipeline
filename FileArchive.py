import os.path
import re
import shutil
import tempfile
import unittest

import Bundle
import LID


class FileArchive(object):
    """An archive containing PDS4 Bundles."""

    def __init__(self, root):
        """Create an archive given a filepath to an existing directory."""
        assert os.path.exists(root) and os.path.isdir(root)
        self.root = root

    def __eq__(self, other):
        return self.root == other.root

    def __str__(self):
        return repr(self.root)

    def __repr__(self):
        return 'FileArchive(%r)' % self.root

    # Verifying parts

    @staticmethod
    def isValidInstrument(inst):
        return inst in ['acs', 'wfc3', 'wfpc2']

    @staticmethod
    def isValidProposal(prop):
        return isinstance(prop, int) and 0 <= prop and prop <= 99999

    @staticmethod
    def isValidVisit(vis):
        try:
            return re.match('\A[a-z0-9][a-z0-9]\Z', vis) is not None
        except:
            return False

    @staticmethod
    def isValidBundleDirBasename(dirname):
        return re.match('\Ahst_[0-9]{5}\Z', dirname) is not None

    @staticmethod
    def isValidCollectionDirBasename(dirname):
        return re.match('\Adata_[a-z0-9]+_', dirname) is not None

    @staticmethod
    def isValidProductDirBasename(dirname):
        return re.match('\Avisit_[a-z0-9]{2}\Z', dirname) is not None

    @staticmethod
    def isHiddenFileBasename(basename):
        return basename[0] == '.'

    # Walking the hierarchy with objects
    def bundles(self):
        """Generate the bundles stored in this archive."""
        for subdir in os.listdir(self.root):
            if re.match(Bundle.Bundle.DIRECTORY_PATTERN, subdir):
                bundleLID = LID.LID('urn:nasa:pds:%s' % subdir)
                yield Bundle.Bundle(self, bundleLID)

    def collections(self):
        """Generate the collections stored in this archive."""
        for b in self.bundles():
            for c in b.collections():
                yield c

    def products(self):
        """Generate the products stored in this archive."""
        for b in self.bundles():
            for c in b.collections():
                for p in c.products():
                    yield p

############################################################


class TestFileArchive(unittest.TestCase):
    def testInit(self):
        with self.assertRaises(Exception):
            FileArchive(None)
        with self.assertRaises(Exception):
            FileArchive("I'm/betting/this/directory/doesn't/exist")

        FileArchive('/')        # guaranteed to exist

        # but try with another directory
        tempdir = tempfile.mkdtemp()
        try:
            FileArchive(tempdir)
        finally:
            shutil.rmtree(tempdir)

    def testStr(self):
        tempdir = tempfile.mkdtemp()
        a = FileArchive(tempdir)
        self.assertEqual(repr(tempdir), str(a))

    def testEq(self):
        tempdir = tempfile.mkdtemp()
        tempdir2 = tempfile.mkdtemp()
        self.assertEquals(FileArchive(tempdir), FileArchive(tempdir))
        self.assertNotEquals(FileArchive(tempdir), FileArchive(tempdir2))

    def testRepr(self):
        tempdir = tempfile.mkdtemp()
        a = FileArchive(tempdir)
        self.assertEqual('FileArchive(%r)' % tempdir, repr(a))

    def testIsValidInstrument(self):
        self.assertTrue(FileArchive.isValidInstrument('wfc3'))
        self.assertTrue(FileArchive.isValidInstrument('wfpc2'))
        self.assertTrue(FileArchive.isValidInstrument('acs'))
        self.assertFalse(FileArchive.isValidInstrument('Acs'))
        self.assertFalse(FileArchive.isValidInstrument('ACS'))
        self.assertFalse(FileArchive.isValidInstrument('ABC'))
        self.assertFalse(FileArchive.isValidInstrument(None))

    def testIsValidProposal(self):
        self.assertFalse(FileArchive.isValidProposal(-1))
        self.assertTrue(FileArchive.isValidProposal(0))
        self.assertTrue(FileArchive.isValidProposal(1))
        self.assertFalse(FileArchive.isValidProposal(100000))
        self.assertFalse(FileArchive.isValidProposal(3.14159265))
        self.assertFalse(FileArchive.isValidProposal('xxx'))
        self.assertFalse(FileArchive.isValidProposal(None))

    def testIsValidVisit(self):
        self.assertTrue(FileArchive.isValidVisit('01'))
        self.assertFalse(FileArchive.isValidVisit('xxx'))
        self.assertFalse(FileArchive.isValidVisit(01))
        self.assertFalse(FileArchive.isValidVisit(None))

if __name__ == '__main__':
    unittest.main()
