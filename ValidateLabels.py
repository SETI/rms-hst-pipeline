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
        for label in get_reduced_products():
            if label:
                failures = xml_schema_failures(None, label) and \
                    schematron_failures(None, label)
                if failures is not None:
                    raise Exception('Validation errors: ' + failures)

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        try:
            return ProductLabelReduction.reduce_product(self,
                                                        archive,
                                                        lid,
                                                        get_reduced_fits_files)
        except IOError:
            # FIXME I'm overlooking IOErrors for now, likely caused by
            # bad FITS headers.  I need to figure out how to handle
            # these.
            pass


class ValidateLabelsReduction(CompositeReduction):
    def __init__(self):
        CompositeReduction.__init__(self, [
                CanMakeValidBundleLabelsReduction(),
                CanMakeValidCollectionLabelsReduction(),
                CanMakeValidProductLabelsReduction()])


if __name__ == '__main__':
    reduction = ValidateLabelsReduction()
    archive = get_any_archive()
    raise_verbosely(lambda: run_reduction(reduction, archive))
    # run_reduction(reduction, archive)
