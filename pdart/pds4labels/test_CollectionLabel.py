from pdart.pds4labels.CollectionLabel import *


def test_make_collection_label():
    from pdart.pds4.Archives import get_any_archive
    arch = get_any_archive()
    b = list(arch.bundles())[0]
    c = list(b.collections())[0]
    make_collection_label(c)
    make_collection_inventory(c)
