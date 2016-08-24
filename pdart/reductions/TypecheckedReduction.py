"""
This module provides typechecking for reductions.

Used during development, *this module is not currently used (as of
2016-08-24).*
"""
from pdart.reductions.Reduction import *


class Typechecks(object):
    """
    A set of methods to typecheck the reductions of an
    :class:`~pdart.pds4.Archive.Archive` or one of its substructures
    (:class:`~pdart.pds4.Bundle.Bundle`,
    :class:`~pdart.pds4.Collection.Collection`, etc.).

    Default behavior is to do nothing.

    Override a method's behavior to raise an exception if a typecheck
    fails.
    """
    def check_is_archive_reduction(self, obj):
        pass

    def check_is_bundle_reduction(self, obj):
        pass

    def check_is_collection_reduction(self, obj):
        pass

    def check_is_product_reduction(self, obj):
        pass

    def check_is_fits_file_reduction(self, obj):
        pass

    def check_is_hdu_reduction(self, obj):
        pass

    def check_is_header_unit_reduction(self, obj):
        pass

    def check_is_data_unit_reduction(self, obj):
        pass


class CompositeTypechecks(Typechecks):
    """
    A :class:`~pdart.reductions.TypecheckedReduction.Typechecks`
    composed of a list of them.  Its behavior is to apply all the base
    :class:`~pdart.reductions.TypecheckedReduction.Typechecks`.
    """
    def __init__(self, typechecks):
        for tc in typechecks:
            assert isinstance(tc, Typechecks)
        self.typechecks = typechecks
        self.size = len(typechecks)

    def check_is_archive_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_archive_reduction(red)

    def check_is_bundle_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_bundle_reduction(red)

    def check_is_collection_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_collection_reduction(red)

    def check_is_product_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_product_reduction(red)

    def check_is_fits_file_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_fits_file_reduction(red)

    def check_is_hdu_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_hdu_reduction(red)

    def check_is_header_unit_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_header_unit_reduction(red)

    def check_is_data_unit_reduction(self, obj):
        assert isinstance(obj, list)
        assert self.size == len(obj)
        for tc, red in zip(self.typechecks, obj):
            tc.check_is_data_unit_reduction(red)


def is_function(func):
    # a -> b
    return hasattr(func, '__call__')


def check_thunk(check, thunk):
    """
    Given a typecheck function that raises an exception if its
    argument is not as expected and a thunk (a no-argument function),
    return a new thunk that produces the same result or raises an
    exception if the result does not pass the typecheck.
    """
    assert is_function(thunk)

    def checked_thunk():
        res = thunk()
        check(res)
        return res
    return checked_thunk


def check_list_thunk(check, thunk):
    """
    Given a typecheck function that raises an exception if its
    argument is not as expected and a thunk (a no-argument function)
    that returns a list, return a new thunk that produces the same
    list of results or raises an exception if any of the results does
    not pass the typecheck.
    """
    assert is_function(thunk)

    def checked_list_thunk():
        res = thunk()
        assert isinstance(res, list)
        for r in res:
            check(r)
        return res
    return checked_list_thunk


class TypecheckedReduction(Reduction):
    """
    A wrapper around a :class:`~pdart.reductions.Reduction.Reduction`
    that acts as a :class:`~pdart.reductions.Reduction.Reduction` with
    a :class:`~pdart.reductions.TypecheckedReduction.Typechecks`
    applied.
    """
    def __init__(self, typechecks, base_reduction):
        assert typechecks
        self.typechecks = typechecks
        assert base_reduction
        self.base_reduction = base_reduction

    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles_tc = \
            check_list_thunk(self.typechecks.check_is_bundle_reduction,
                             get_reduced_bundles)
        res = self.base_reduction.reduce_archive(archive_root,
                                                 get_reduced_bundles_tc)
        self.typechecks.check_is_archive_reduction(res)
        return res

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections_tc = \
            check_list_thunk(self.typechecks.check_is_collection_reduction,
                             get_reduced_collections)
        res = self.base_reduction.reduce_bundle(archive,
                                                lid,
                                                get_reduced_collections_tc)
        self.typechecks.check_is_bundle_reduction(res)
        return res

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products_tc = \
            check_list_thunk(self.typechecks.check_is_product_reduction,
                             get_reduced_products)
        res = self.base_reduction.reduce_collection(archive,
                                                    lid,
                                                    get_reduced_products_tc)
        self.typechecks.check_is_collection_reduction(res)
        return res

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        get_reduced_fits_files_tc = \
            check_list_thunk(self.typechecks.check_is_fits_file_reduction,
                             get_reduced_fits_files)
        res = self.base_reduction.reduce_product(archive,
                                                 lid,
                                                 get_reduced_fits_files_tc)
        self.typechecks.check_is_product_reduction(res)
        return res

    def reduce_fits_file(self, file, get_reduced_hdus):
        get_reduced_hdus_tc = \
            check_list_thunk(self.typechecks.check_is_hdu_reduction,
                             get_reduced_hdus)
        res = self.base_reduction.reduce_fits_file(file,
                                                   get_reduced_hdus_tc)
        self.typechecks.check_is_fits_file_reduction(res)
        return res

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        get_reduced_header_unit_tc = \
            check_thunk(self.typechecks.check_is_header_unit_reduction,
                        get_reduced_header_unit)
        get_reduced_data_unit_tc = \
            check_thunk(self.typechecks.check_is_data_unit_reduction,
                        get_reduced_data_unit)
        res = self.base_reduction.reduce_hdu(n,
                                             hdu,
                                             get_reduced_header_unit_tc,
                                             get_reduced_data_unit_tc)
        self.typechecks.check_is_hdu_reduction(res)
        return res

    def reduce_header_unit(self, n, header_unit):
        res = self.base_reduction.reduce_header_unit(n,
                                                     header_unit)
        self.typechecks.check_is_header_unit_reduction(res)
        return res

    def reduce_data_unit(self, n, data_unit):
        res = self.base_reduction.reduce_data_unit(n,
                                                   data_unit)
        self.typechecks.check_is_data_unit_reduction(res)
        return res


class TypecheckedReductionRunner(object):
    """
    A class to run a :class:`~pdart.reductions.Reduction.Reduction` on
    an :class:`~pdart.pds4.Archive.Archive` or one of its
    substructures (:class:`~pdart.pds4.Bundle.Bundle`,
    :class:`~pdart.pds4.Collection.Collection`, etc.) while
    typechecking the reductions.
    """
    def __init__(self, typechecks, base_runner):
        assert typechecks
        self.typechecks = typechecks
        assert base_runner
        self.base_runner = base_runner

    def run_archive(self, reduction, archive):
        res = self.base_runner.run_archive(reduction, archive)
        self.typechecks.check_is_archive_reduction(res)
        return res

    def run_bundle(self, reduction, bundle):
        res = self.base_runner.run_bundle(reduction, bundle)
        self.typechecks.check_is_bundle_reduction(res)
        return res

    def run_collection(self, reduction, collection):
        res = self.base_runner.run_collection(reduction, collection)
        self.typechecks.check_is_collection_reduction(res)
        return res

    def run_product(self, reduction, product):
        res = self.base_runner.run_product(reduction, product)
        self.typechecks.check_is_product_reduction(res)
        return res

    def run_fits_file(self, reduction, file):
        res = self.base_runner.run_fits_file(reduction, file)
        self.typechecks.check_is_fits_file_reduction(res)
        return res

    def run_hdu(self, reduction, n, hdu):
        res = self.base_runner.run_hdu(reduction, n, hdu)
        self.typechecks.check_is_hdu_reduction(res)
        return res

    def run_header_unit(self, reduction, n, header_unit):
        res = self.base_runner.run_header_unit(reduction,
                                               n,
                                               header_unit)
        self.typechecks.check_is_header_unit_reduction(res)
        return res

    def run_data_unit(self, reduction, n, data_unit):
        res = self.base_runner.run_data_unit(reduction, n, data_unit)
        self.typechecks.check_is_data_unit_reduction(res)
        return res
