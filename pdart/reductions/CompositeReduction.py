"""
This module provides functionality for composing
:class:`~pdart.reductions.Reduction.Reduction` s.
"""
from pdart.reductions.Reduction import *

from typing import Any, Callable, Sequence


def indexed2(func):
    # type: (Callable[[], Sequence[Any]]) -> Callable[[int], Callable[[], Any]]
    """
    A version of :func:`indexed` used for header and data units
    because there is only ever one element in them.  Most of the
    documentation for :func:`indexed` should apply.
    """
    cache = {'is_set': False, 'value': None}

    def cache_func_result():
        if not cache['is_set']:
            cache['is_set'] = True
            # The original function
            cache['value'] = func()

    def indexed2_func(i):
        def thunk():
            cache_func_result()
            value = cache['value']
            try:
                return value[i]
            except IndexError:
                raise IndexError(
                    'list index %d out of range for %s in indexed2()' %
                    (i, value))

        return thunk

    return indexed2_func


def indexed(func):
    # type: (Callable[[], Sequence[Any]]) -> Callable[[int], Callable[[], Any]]
    """
    Convert a thunk (no-parameter function) returning a tuple into a
    function taking an index and returning a thunk that gives the
    tuple element at that index.

    This function is used to minimize calculation: we only want to
    reduce lower levels if we need to, but not all reductions in a
    :class:`CompositeReduction` might need to go equally deep.  By
    indexing, we delay the calculation unless it's needed, and we also
    cache the result of lower calculations so they aren't done
    multiple times.  (You don't have to understand the implementation
    unless you've found a bug in it.)

    The ``get_reduced_xxx()`` argument passed to
    :class:`pdart.reductions.CompositeReduction` 's ``reduce_yyy()``
    methods is a thunk that returns a list (of the same length as the
    list of ``xxx`` substructures) of lists (of the same length as the
    list of reductions in the composite).

    When you give a reduction index to the result thunk, you get a
    thunk that returns a list with all the reduced substructures that
    should go to that reduction.

    The elements of the result of original thunk are accessed
    ``res[sub][red]``.  Then ``indexed(f)(r)()`` returns ``[res[s][r]
    for s in range(0, len(res)]``, or equivalently, ``[res_elmt[r] for
    res_elmt in res]``, or equivalently, ``transpose(res)[r]``.

    If Python had static types and a compiler that used them, I
    wouldn't have needed to write the previous paragraph.
    """
    cache = {'is_set': False, 'value': None}

    def transpose(list_of_lists):
        try:
            return map(list, zip(*list_of_lists))
        except TypeError as e:
            raise TypeError('%s; list_of_lists=%s' % (e, list_of_lists))

    def cache_func_result():
        if not cache['is_set']:
            cache['is_set'] = True
            # The original function
            cache['value'] = transpose(func())

    def indexed_func(i):
        def thunk():
            cache_func_result()
            value = cache['value']
            try:
                return value[i]
            except IndexError:
                raise IndexError(
                    'list index %d out of range for %s in indexed()' %
                    (i, value))

        return thunk

    return indexed_func


class CompositeReduction(Reduction):
    """
    A :class:`~pdart.reductions.Reduction.Reduction` made from
    combining :class:`~pdart.reductions.Reduction.Reduction` s.
    Results consist of lists of result from the component
    :class:`~pdart.reductions.Reduction.Reduction` s but the recursion
    is only performed once.

    NOTE: all of the ``reduce_xxx()`` methods here return lists of
    length *r*, where *r* is the length of ``self.reductions``.  All
    the ``get_reduced_xxx()`` functions passed as arguments return
    lists of length *s*, where *s* is the number of subcomponents.

    The indices *i* passed to the ``get_reduced_xxx_indexed()``
    functions are bound by 0 <= *i* < *r*.
    """
    def __init__(self, reductions):
        assert reductions
        assert isinstance(reductions, list)
        self.reductions = reductions
        self.count = len(reductions)

    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles_indexed = indexed(get_reduced_bundles)

        return [r.reduce_archive(archive_root,
                                 get_reduced_bundles_indexed(i))
                for i, r in enumerate(self.reductions)]

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections_indexed = indexed(get_reduced_collections)
        return [r.reduce_bundle(archive, lid,
                                get_reduced_collections_indexed(i))
                for i, r in enumerate(self.reductions)]

    def reduce_collection(self, archive, lid, get_reduced_products):
        get_reduced_products_indexed = indexed(get_reduced_products)
        return [r.reduce_collection(archive, lid,
                                    get_reduced_products_indexed(i))
                for i, r in enumerate(self.reductions)]

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        get_reduced_fits_files_indexed = indexed(get_reduced_fits_files)
        return [r.reduce_product(archive, lid,
                                 get_reduced_fits_files_indexed(i))
                for i, r in enumerate(self.reductions)]

    def reduce_fits_file(self, file, get_reduced_hdus):
        get_reduced_hdus_indexed = indexed(get_reduced_hdus)
        return [r.reduce_fits_file(file, get_reduced_hdus_indexed(i))
                for i, r in enumerate(self.reductions)]

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        get_reduced_header_unit_indexed = indexed2(get_reduced_header_unit)
        get_reduced_data_unit_indexed = indexed2(get_reduced_data_unit)
        return [r.reduce_hdu(n, hdu,
                             get_reduced_header_unit_indexed(i),
                             get_reduced_data_unit_indexed(i))
                for i, r in enumerate(self.reductions)]

    def reduce_header_unit(self, n, header_unit):
        return [r.reduce_header_unit(n, header_unit) for r in self.reductions]

    def reduce_data_unit(self, n, data_unit):
        return [r.reduce_data_unit(n, data_unit) for r in self.reductions]


# def composite_reduction_type(dicts):
#     KEYS = ['archive', 'bundle', 'collection', 'product',
#             'fits_file', 'hdu', 'header_unit', 'data_unit']
#     res = {}
#     for k in KEYS:
#         vs = [d[k] for d in dicts]
#         res[k] = '(' + ', '.join(vs) + ')'
#     return res
