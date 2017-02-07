"""
**SCRIPT:** Run through the archive creating and validating bundle,
collection and product labels.  If it fails, print the combined
exception to stdout.  If it succeeds, do nothing.  The labels are not
saved to disk.
"""

from pdart.pds4.Archives import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.FitsProductLabel import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.InstrumentationReductions import *
from pdart.rules.Combinators import *


class CanMakeValidBundleLabelsReduction(BundleLabelReduction):
    """
    Verifies that within the archive, valid bundle labels can be
    built.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        for label in get_reduced_bundles():
            failures = xml_schema_failures(None, label) or \
                schematron_failures(None, label)
            if failures is not None:
                raise Exception('Validation errors: ' + failures)


class CanMakeValidCollectionLabelsReduction(CollectionLabelReduction):
    """
    Verifies that within the archive, valid collection labels can be
    built.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        for label in get_reduced_collections():
            failures = xml_schema_failures(None, label) or \
                schematron_failures(None, label)
            if failures is not None:
                raise Exception('Validation errors: ' + failures)


class CanMakeValidProductLabelsReduction(FitsProductLabelReduction):
    """
    Verifies that within the archive, valid FITS product labels can be
    built.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        for prod in get_reduced_products():
            if prod:
                lid, label = prod
                failures = xml_schema_failures(None, label) or \
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


class ValidateLabelsReduction(CompositeReduction):
    """
    Combines previous validation reductions.
    """
    def __init__(self):
        # type: () -> None
        CompositeReduction.__init__(self, [
                LogProductsReduction(),
                CanMakeValidBundleLabelsReduction(),
                CanMakeValidCollectionLabelsReduction(),
                CanMakeValidProductLabelsReduction()])


def run():
    # type: () -> None
    reduction = ValidateLabelsReduction()
    archive = get_any_archive()
    raise_verbosely(lambda: run_reduction(reduction, archive))
    # run_reduction(reduction, archive)

if __name__ == '__main__':
    run()
