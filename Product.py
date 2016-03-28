import os.path
import re
import shutil
import tempfile
import unittest

import ArchiveComponent
import ArchiveFile
import Bundle
import Collection
import FileArchive
import LID


def _find_filepath_under_dir(dir, filename):
    """
    Find a file by name in a directory or one of its subdirectories
    and return the absolute filepath.  Assume the directory path is
    absolute and that only one file with that name exists under the
    directory.
    """
    # TODO: we can get the visit number from the filename, so no need
    # to review all the directories (or at least we can try the most
    # likely one first).
    for (dirpath, dirnames, filenames) in os.walk(dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)


class Product(ArchiveComponent.ArchiveComponent):
    """A PDS4 Product."""

    VISIT_DIRECTORY_PATTERN = r'\Avisit_([a-z0-9]{2})\Z'

    def __init__(self, arch, lid):
        """
        Create a Product given the archive it lives in and its LID.
        """
        assert lid.is_product_lid()
        super(Product, self).__init__(arch, lid)

    def __repr__(self):
        return 'Product(%r, %r)' % (self.archive, self.lid)

    def absolute_filepath(self):
        """Return the absolute filepath to the product file."""
        collection_fp = self.collection().absolute_filepath()
        res = _find_filepath_under_dir(collection_fp,
                                       self.lid.product_id + '.fits')
        assert res, ('Product.absolute_filepath(%r) = %r '
                     'where collection_fp = %r' % (self, res, collection_fp))
        return res

    def visit(self):
        """
        Return the visit code for this product.  It is calculated from
        the product's filepath
        """
        fp = self.absolute_filepath()
        visit_fp = os.path.basename(os.path.dirname(fp))
        return re.match(Product.VISIT_DIRECTORY_PATTERN, visit_fp).group(1)

    def files(self):
        basename = os.path.basename(self.absolute_filepath())
        return [ArchiveFile.ArchiveFile(self, basename)]

    def absolute_filepath_is_directory(self):
        return False

    def collection(self):
        """Return the collection this product belongs to."""
        return Collection.Collection(self.archive, self.lid.parent_lid())

    def bundle(self):
        """Return the bundle this product belongs to."""
        return Bundle.Bundle(self.archive, self.lid.parent_lid().parent_lid())


class TestProduct(unittest.TestCase):
    def test_find_filepath_under_dir(self):
        # This is only a sanity test since we assume os.walk() works
        # correctly.
        tempdir = tempfile.mkdtemp()
        try:
            dst_dir = os.path.join(tempdir, 'a/b/c/d/e')
            os.makedirs(dst_dir)
            filepath = os.path.join(dst_dir, 'foo.fits')
            with open(filepath, 'w') as f:
                f.write('Hi!\n')
            self.assertEqual(filepath,
                             _find_filepath_under_dir(dst_dir, 'foo.fits'))
        finally:
            shutil.rmtree(tempdir)

    def test_visit(self):
        tempdir = tempfile.mkdtemp()
        try:
            archive = FileArchive.FileArchive(tempdir)
            visit_dir = os.path.join(tempdir,
                                     'hst_99999/data_acs_xxx/visit_23/')
            os.makedirs(visit_dir)
            filepath = os.path.join(visit_dir, 'foo_xxx.fits')
            with open(filepath, 'w') as f:
                f.write('Hi!\n')
            lid = LID.LID('urn:nasa:pds:hst_99999:data_acs_xxx:foo_xxx')
            product = Product(archive, lid)
            self.assertEqual('23', product.visit())
        finally:
            shutil.rmtree(tempdir)


if __name__ == '__main__':
    unittest.main()
