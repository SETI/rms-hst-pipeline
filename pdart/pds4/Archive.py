import os.path
import re
import shutil
import tempfile
import unittest

import pdart.pds4.Bundle
import pdart.pds4.LID


class Archive(object):
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
        return 'Archive(%r)' % self.root

    # Verifying parts

    @staticmethod
    def is_valid_instrument(inst):
        return inst in ['acs', 'wfc3', 'wfpc2']

    @staticmethod
    def is_valid_proposal(prop):
        return isinstance(prop, int) and 0 <= prop and prop <= 99999

    @staticmethod
    def is_valid_visit(vis):
        try:
            return re.match(r'\A[a-z0-9][a-z0-9]\Z', vis) is not None
        except:
            return False

    @staticmethod
    def is_valid_bundle_dir_basename(dirname):
        return re.match(r'\Ahst_[0-9]{5}\Z', dirname) is not None

    @staticmethod
    def is_valid_collection_dir_basename(dirname):
        return re.match(r'\Adata_[a-z0-9]+_', dirname) is not None

    @staticmethod
    def is_valid_product_dir_basename(dirname):
        return re.match(r'\Avisit_[a-z0-9]{2}\Z', dirname) is not None

    @staticmethod
    def is_hidden_file_basename(basename):
        return basename[0] == '.'

    # Walking the hierarchy with objects
    def bundles(self):
        """Generate the bundles stored in this archive."""
        for subdir in os.listdir(self.root):
            if re.match(pdart.pds4.Bundle.Bundle.DIRECTORY_PATTERN, subdir):
                bundle_lid = pdart.pds4.LID.LID('urn:nasa:pds:%s' % subdir)
                yield pdart.pds4.Bundle.Bundle(self, bundle_lid)

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


class TestArchive(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(Exception):
            Archive(None)
        with self.assertRaises(Exception):
            Archive("I'm/betting/this/directory/doesn't/exist")

        Archive('/')        # guaranteed to exist

        # but try with another directory
        tempdir = tempfile.mkdtemp()
        try:
            Archive(tempdir)
        finally:
            shutil.rmtree(tempdir)

    def test_str(self):
        tempdir = tempfile.mkdtemp()
        a = Archive(tempdir)
        self.assertEqual(repr(tempdir), str(a))

    def test_eq(self):
        tempdir = tempfile.mkdtemp()
        tempdir2 = tempfile.mkdtemp()
        self.assertEquals(Archive(tempdir), Archive(tempdir))
        self.assertNotEquals(Archive(tempdir), Archive(tempdir2))

    def test_repr(self):
        tempdir = tempfile.mkdtemp()
        a = Archive(tempdir)
        self.assertEqual('Archive(%r)' % tempdir, repr(a))

    def test_is_valid_instrument(self):
        self.assertTrue(Archive.is_valid_instrument('wfc3'))
        self.assertTrue(Archive.is_valid_instrument('wfpc2'))
        self.assertTrue(Archive.is_valid_instrument('acs'))
        self.assertFalse(Archive.is_valid_instrument('Acs'))
        self.assertFalse(Archive.is_valid_instrument('ACS'))
        self.assertFalse(Archive.is_valid_instrument('ABC'))
        self.assertFalse(Archive.is_valid_instrument(None))

    def test_is_valid_proposal(self):
        self.assertFalse(Archive.is_valid_proposal(-1))
        self.assertTrue(Archive.is_valid_proposal(0))
        self.assertTrue(Archive.is_valid_proposal(1))
        self.assertFalse(Archive.is_valid_proposal(100000))
        self.assertFalse(Archive.is_valid_proposal(3.14159265))
        self.assertFalse(Archive.is_valid_proposal('xxx'))
        self.assertFalse(Archive.is_valid_proposal(None))

    def test_is_valid_visit(self):
        self.assertTrue(Archive.is_valid_visit('01'))
        self.assertFalse(Archive.is_valid_visit('xxx'))
        self.assertFalse(Archive.is_valid_visit(01))
        self.assertFalse(Archive.is_valid_visit(None))

if __name__ == '__main__':
    unittest.main()
