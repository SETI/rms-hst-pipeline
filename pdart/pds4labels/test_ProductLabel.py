from pdart.pds4labels.ProductLabel import *


def test_make_product_label():
    from pdart.pds4.Archives import get_any_archive
    arch = get_any_archive()
    b = list(arch.bundles())[0]
    c = list(b.collections())[0]
    p = list(c.products())[0]
    print make_product_label(p, True)
    assert False
