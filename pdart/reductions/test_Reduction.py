from pdart.exceptions.ExceptionInfo import CalculationException
from pdart.pds4.Archives import *
from pdart.reductions.Reduction import *


class TestRecursiveReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        return 1 + sum(get_reduced_bundles())

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        return 1 + sum(get_reduced_collections())

    def reduce_collection(self, archive, lid, get_reduced_products):
        return 1 + sum(get_reduced_products())

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        return 1 + sum(get_reduced_fits_files())

    def reduce_fits_file(self, file, get_reduced_hdus):
        # We don't go any deeper because it takes too long for a unit
        # test.  (In fact, it might be too long with the full
        # archive.)
        return 1


def test_reductions():
    arch = get_any_archive()
    try:
        count = run_reduction(TestRecursiveReduction(), arch)
        assert count == 137567
    except CalculationException as ce:
        print ce.exception_info.to_pretty_xml()
        raise


if False:
    def test_type_documentation():
        d = {'archive': 'None',
             'bundle': 'None',
             'collection': 'None',
             'product': 'ProductLabel',
             'fits_file': 'dict',
             'hdu': 'dict',
             'header_unit': 'None',
             'data_unit': 'None'}
        print reduction_type_documentation(d)
        assert False
