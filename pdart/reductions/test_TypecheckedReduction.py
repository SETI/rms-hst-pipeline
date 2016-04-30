from pdart.exceptions.ExceptionInfo import *
from pdart.reductions.TypecheckedReduction import *

if False:
    def test():
        class NoneTypechecks(object):
            def check_is_archive_reduction(self, obj):
                assert obj is None

            def check_is_bundle_reduction(self, obj):
                assert obj is None

            def check_is_collection_reduction(self, obj):
                assert obj is None

            def check_is_product_reduction(self, obj):
                assert obj is None

            def check_is_fits_file_reduction(self, obj):
                assert obj is None

            def check_is_hdu_reduction(self, obj):
                assert obj is None

            def check_is_header_unit_reduction(self, obj):
                assert obj is None

            def check_is_data_unit_reduction(self, obj):
                assert obj is None

        class DeepReduction(Reduction):
            def reduce_archive(self, archive_root, get_reduced_bundles):
                get_reduced_bundles()
                return None

            def reduce_bundle(self, archive, lid, get_reduced_collections):
                get_reduced_collections()
                return None

            def reduce_collection(self, archive, lid, get_reduced_products):
                get_reduced_products()
                return None

            def reduce_product(self, archive, lid, get_reduced_fits_files):
                get_reduced_fits_files()
                return None

            def reduce_fits_file(self, file, get_reduced_hdus):
                get_reduced_hdus()
                return None

            def reduce_hdu(self, n, hdu,
                           get_reduced_header_unit,
                           get_reduced_data_unit):
                get_reduced_header_unit()
                get_reduced_data_unit()
                return None

            def reduce_header_unit(self, n, header_unit):
                return None

            def reduce_data_unit(self, n, data_unit):
                return None

        tc = NoneTypechecks()
        red = TypecheckedReduction(tc, DeepReduction())
        runner = TypecheckedReductionRunner(tc, DefaultReductionRunner())
        from pdart.pds4.Archives import get_any_archive
        arch = get_any_archive()

        try:
            runner.run_archive(red, arch)
        except CalculationException as ce:
            print ce.exception_info.to_pretty_xml()
            raise
