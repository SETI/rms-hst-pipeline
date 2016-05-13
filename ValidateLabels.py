from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *
from pdart.reductions.CompositeReduction import *


class CanMakeValidBundleLabelsReduction(BundleLabelReduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        for label in get_reduced_bundles():
            failures = xml_schema_failures(None, label) and \
                schematron_failures(None, label)
            if failures is not None:
                raise Exception('Validation errors: ' + failures)


class CanMakeValidCollectionLabelsReduction(CollectionLabelReduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        for label in get_reduced_collections():
            failures = xml_schema_failures(None, label) and \
                schematron_failures(None, label)
            if failures is not None:
                raise Exception('Validation errors: ' + failures)


class CanMakeValidProductLabelsReduction(ProductLabelReduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        for prod in get_reduced_products():
            if prod:
                lid, label = prod
                failures = xml_schema_failures(None, label) and \
                    schematron_failures(None, label)
                if failures is not None:
                    raise Exception('Validation errors in %s: %s' %
                                    (lid, failures))

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        reduced_fits_files = get_reduced_fits_files()
        for fits_file in reduced_fits_files:
            if fits_file is None:
                return None

        def get_reduced_fits_files_no_fail():
            return reduced_fits_files

        return (lid,
                ProductLabelReduction.reduce_product(
                    self,
                    archive,
                    lid,
                    get_reduced_fits_files_no_fail))

    def reduce_fits_file(self, file, get_reduced_hdus):
        try:
            reduced_hdus = get_reduced_hdus()
        except IOError:
            return None

        def get_reduced_hdus_no_fail():
            return reduced_hdus

        return ProductLabelReduction.reduce_fits_file(self,
                                                      file,
                                                      get_reduced_hdus_no_fail)


class LogProductsReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_products):
        print str(lid)


class ValidateLabelsReduction(CompositeReduction):
    def __init__(self):
        CompositeReduction.__init__(self, [
                LogProductsReduction(),
                CanMakeValidBundleLabelsReduction(),
                CanMakeValidCollectionLabelsReduction(),
                CanMakeValidProductLabelsReduction()])


if __name__ == '__main__':
    reduction = ValidateLabelsReduction()
    archive = get_any_archive()
    raise_verbosely(lambda: run_reduction(reduction, archive))
    # run_reduction(reduction, archive)
