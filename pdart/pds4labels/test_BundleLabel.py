from pdart.pds4labels.BundleLabel import *


def test_make_bundle_label():
    from pdart.pds4.Archives import get_any_archive
    arch = get_any_archive()
    b = list(arch.bundles())[0]
    make_bundle_label(b, True)
