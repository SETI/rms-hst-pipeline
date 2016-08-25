"""
**SCRIPT:** Run through the archive checking for certain invariant
properties.  If any of the properties fail to hold, print the combined
exception to stdout.  If it succeeds, do nothing.
"""

from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4.Bundle import *
from pdart.pds4.Collection import *
from pdart.pds4.HstFilename import *
from pdart.reductions.CompositeReduction import *


def _unions(sets):
    """Unions a list of sets."""
    res = set()
    for s in sets:
        res |= s
    return res


class BundleContainsOneSingleHstInternalProposalIdReduction(Reduction):
    """
    Verifies that within the archive, each bundle contains exactly one
    HST proposal id.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        collections = _unions(get_reduced_collections())
        assert 1 == len(collections)

    def reduce_collection(self, archive, lid, get_reduced_products):
        return _unions(get_reduced_products())

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        return set(get_reduced_fits_files())

    def reduce_fits_file(self, file, get_reduced_hdus):
        return HstFilename(file.full_filepath()).hst_internal_proposal_id()


class ProductFilesHaveBundleProposalIdReduction(Reduction):
    """
    Verifies that within the archive, each FITS file's ``PROPOSID``
    field matches the proposal id of the containing bundle.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        get_reduced_fits_files()

    def reduce_fits_file(self, file, get_reduced_hdus):
        # returns int
        try:
            proposid = pyfits.getval(file.full_filepath(), 'PROPOSID')
        except (IOError, KeyError):
            proposid = None

        if proposid is not None:
            product = file.component
            collection = product.collection()
            collection_suffix = collection.suffix()
            if collection_suffix == 'lrc':
                assert 0 == proposid
            else:
                bundle = product.bundle()
                assert bundle.proposal_id() == proposid


class ProductFilesHaveCollectionSuffixReduction(Reduction):
    """
    Verifies that within the archive, the suffix embedded in each
    product FITS filename matches the suffix embedded in its
    containing collection directory name.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        reduced_products = _unions(get_reduced_products())
        collection = Collection(archive, lid)
        assert set([collection.suffix()]) == reduced_products

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # returns set(string)
        reduced_fits_files = get_reduced_fits_files()
        return set(reduced_fits_files)

    def reduce_fits_file(self, file, get_reduced_hdus):
        # returns string suffix
        return HstFilename(file.basename).suffix()


class ProductFilesHaveProductVisitReduction(Reduction):
    """
    Verifies that within the archive, the visit embedded in each
    product FITS filename matches the visit embedded in its
    containing visit directory name.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products()

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        get_reduced_fits_files()

    def reduce_fits_file(self, file, get_reduced_hdus):
        file_visit = HstFilename(file.full_filepath()).visit()
        product_visit = file.component.visit()
        assert product_visit == file_visit


class ValidationReduction(CompositeReduction):
    """
    Combines previous validation reductions.
    """
    def __init__(self):
        CompositeReduction.__init__(self, [
                BundleContainsOneSingleHstInternalProposalIdReduction(),
                ProductFilesHaveBundleProposalIdReduction(),
                ProductFilesHaveCollectionSuffixReduction(),
                ProductFilesHaveProductVisitReduction()])


if __name__ == '__main__':
    reduction = ValidationReduction()
    archive = get_any_archive()
    raise_verbosely(lambda: run_reduction(reduction, archive))
    # run_reduction(reduction, archive)
