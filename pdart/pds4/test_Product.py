import os.path
import unittest

from pdart.pds4.Archives import get_any_archive
from pdart.pds4.Bundle import *
from pdart.pds4.Collection import *
from pdart.pds4.HstFilename import *
from pdart.pds4.Product import *
from pdart.pds4.LID import LID


class TestProduct(unittest.TestCase):
    def test_init(self):
        # type: () -> None
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection:uproduct')
        p = Product(arch, lid)
        self.assertEquals(lid, p.lid)

        # check that creation of bundle fails with collection LID
        lid = LID('urn:nasa:pds:bundle:collection')
        try:
            Collection(arch, lid)
            self.assertTrue(False)
        except Exception:
            pass

    def test_absolute_filepath(self):
        # type: () -> None
        arch = get_any_archive()
        for p in arch.products():
            lid = p.lid
            visit = p.visit()

            actual_fp = p.absolute_filepath()
            expected_fps = [os.path.join(arch.root,
                                         lid.bundle_id,
                                         lid.collection_id,
                                         ('visit_%s' % visit),
                                         lid.product_id + ext)
                            for ext in Product.FILE_EXTS]

            assert actual_fp in expected_fps
            # We only check the first product
            return

    def test_bundle(self):
        # type: () -> None
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection:uproduct')
        p = Product(arch, lid)
        self.assertEquals(Bundle(arch, LID('urn:nasa:pds:bundle')),
                          p.bundle())

    def test_collection(self):
        # type: () -> None
        arch = get_any_archive()
        lid = LID('urn:nasa:pds:bundle:collection:uproduct')
        p = Product(arch, lid)
        self.assertEquals(Collection(arch,
                                     LID('urn:nasa:pds:bundle:collection')),
                          p.collection())
