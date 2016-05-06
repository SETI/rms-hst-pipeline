from pdart.exceptions.ExceptionInfo import CalculationException
from pdart.pds4.Archives import *
from pdart.reductions.Reduction import *


def _not_all_same(lst):
    return len(lst) == 1 or len(set(lst)) != 1


class CheckBundleReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        reduced_bundles = get_reduced_bundles()
        assert _not_all_same(reduced_bundles)

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        return lid


class CheckCollectionReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        reduced_collections = get_reduced_collections()
        assert _not_all_same(reduced_collections)

    def reduce_collection(self, archive, lid, get_reduced_products):
        return lid


class CheckProductReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        reduced_products = get_reduced_products()
        assert _not_all_same(reduced_products)

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        return lid


class CheckFileReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        reduced_fits_files = get_reduced_fits_files()
        assert _not_all_same(reduced_fits_files)

    def reduce_fits_file(self, file, get_reduced_hdus):
        return file.full_filepath()


class CheckHduReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        get_reduced_fits_files()

    def reduce_fits_file(self, file, get_reduced_hdus):
        reduced_hdus = get_reduced_hdus()
        assert _not_all_same(reduced_hdus)

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        return n


def test_check_reduction():
    arch = get_any_archive()
    run_reduction(CheckBundleReduction(), arch)
    run_reduction(CheckCollectionReduction(), arch)
    run_reduction(CheckProductReduction(), arch)
    run_reduction(CheckFileReduction(), arch)
    if False:
        # This one is slow because it has to open and parse each
        # FITS file.
        run_reduction(CheckHduReduction(), arch)


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
        # We don't go any deeper because it requires opening and
        # parsing the FITS file and the cost of opening all the files
        # would make it run too slowly for a unit test.
        return 1


def test_reductions():
    arch = get_any_archive()
    try:
        count = run_reduction(TestRecursiveReduction(), arch)
        # FIXME One value for mini archive on dev machine and one for
        # full archive on test machine.
        assert count in [9529]
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
