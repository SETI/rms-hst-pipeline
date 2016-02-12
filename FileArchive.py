import os.path
import re

import Bundle
import LID

class FileArchive:
    def __init__(self, root):
	assert os.path.exists(root)
	self.root = root

    def __eq__(self, other):
        return self.root == other.root

    def __str__(self):
	return repr(self.root)

    def __repr__(self):
	return 'FileArchive(%s)' % repr(self.root)

    #### Verifying parts

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

    #### Walking the hierarchy

    def walkComp(self, start=None):
        if start == None:
            start = self.root
        rootSegs = self.root.split('/')
        startSegs = start.split('/')
        level = len(startSegs) - len(rootSegs)
        if level == 0:
            # start is the root; walk bundles
            for subdir in os.listdir(start):
                bundleDir = os.path.join(self.root, subdir)
                if (FileArchive.isValidBundleDirBasename(subdir) 
                    and os.path.isdir(bundleDir)):
                    yield bundleDir
        elif level == 1:
            # start is a bundle; walk collections
            for subdir in os.listdir(start):
                collectionDir = os.path.join(start, subdir)
                if (FileArchive.isValidCollectionDirBasename(subdir)
                    and os.path.isdir(collectionDir)):
                    yield collectionDir
        elif level == 2:
            # start is a collection; walk products
            for subdir in os.listdir(start):
                productDir = os.path.join(start, subdir)
                if (FileArchive.isValidProductDirBasename(subdir)
                    and os.path.isdir(productDir)):
                    yield productDir
        elif level == 3:
            # start is a product; walk files
            for f in os.listdir(start):
                file = os.path.join(start, f)
                if (not FileArchive.isHiddenFileBasename(f)
                    and os.path.isfile(file)):
                    yield file
        else:
            # it's a bug
            assert false

    def walkBundleDirectories(self):
        for bundleDir in self.walkComp():
            yield bundleDir

    def walkCollectionDirectories(self):
        for bundleDir in self.walkBundleDirectories():
            for collectionDir in self.walkComp(bundleDir):
                yield collectionDir

    def walkProductDirectories(self):
        for collectionDir in self.walkCollectionDirectories():
            for subdir in os.listdir(collectionDir):
                productDir = os.path.join(collectionDir, subdir)
                if (FileArchive.isValidProductDirBasename(subdir)
                    and os.path.isdir(productDir)):
                    yield productDir

    # Walking the hierarchy with objects
    def bundles(self):
        for subdir in os.listdir(self.root):
            if re.match(Bundle.Bundle.DIRECTORY_PATTERN, subdir):
                bundleLID = LID.LID('urn:nasa:pds:%s' % subdir)
                yield Bundle.Bundle(self, bundleLID)
        

############################################################

import shutil
import tempfile
import unittest

class TestFileArchive(unittest.TestCase):
    def testInit(self):
	with self.assertRaises(Exception):
	    FileArchive(None)
	with self.assertRaises(Exception):
	    FileArchive("I'm/betting/this/directory/doesn't/exist")

	FileArchive('/')	# guaranteed to exist

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
	self.assertEqual('FileArchive(%s)' % repr(tempdir), repr(a))

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
