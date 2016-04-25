from pdart.pds4labels.ProductLabel import *
from pdart.exceptions.ExceptionInfo import CalculationException
from pdart.xml.Pretty import pretty_print


def test_make_product_label():
    from pdart.pds4.Archives import get_any_archive
    arch = get_any_archive()
    b = list(arch.bundles())[-1]
    c = list(b.collections())[-1]
    p = list(c.products())[-1]

    # Verify the label against its XML Schema and Schematron.
    VERIFY = True

    # Be verbose if you want to inspect the full results when running
    # the test, whether it's valid or not.
    FAIL_WITH_VERBOSE_RESULTS = False

    if FAIL_WITH_VERBOSE_RESULTS:
        try:
            print pretty_print(make_product_label(p, VERIFY))
        except CalculationException as ce:
            print ce.exception_info.to_pretty_xml()
        assert False
    else:
        make_product_label(p, VERIFY)
