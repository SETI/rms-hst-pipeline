from pdart.reductions.Reduction import *


def indexed2(func):
    # FIXME Document and clean this up.  We have this version for
    # header and data units because there are only ever one element in
    # them.
    cache = {'is_set': False, 'value': None}

    def store_func_result():
        if not cache['is_set']:
            cache['is_set'] = True
            # The original function
            cache['value'] = func()

    def i_th(elmt, i):
        try:
            return elmt[i]
        except IndexError:
            print 'tried to index %s at %d' % (elmt, i)
            raise

    def indexed2_func(i):
        def thunk():
            store_func_result()
            return i_th(cache['value'], i)
        return thunk

    return indexed2_func


def indexed(func):
    """
    Convert a thunk to a function returning thunks.

    The get_reduced_xxx argument passed to :class:`CompositeReduction`
    reduce_yyy() methods is a thunk that returns a list (of the same
    length as the list of xxx substructures) of lists (of the same
    length as the list of reductions in the composite).

    When you give a reduction index to the result thunk, you get a
    thunk that returns a list with all the reduced substructures that
    should go to that reduction.

    The elements of the result of original thunk are accessed
    res[sub][red].  Then indexed(f)(r)() returns [res[s][r] for s in
    range(0, len(res)], or equivalently, [res_elmt[r] for res_elmt in
    res], or equivalently, transpose(res)[r].
    """
    cache = {'is_set': False, 'value': None}

    def transpose(list_of_lists): return map(list, zip(*list_of_lists))

    def store_func_result():
        if not cache['is_set']:
            cache['is_set'] = True
            # The original function
            cache['value'] = transpose(func())

    def i_th(elmt, i):
        try:
            return elmt[i]
        except IndexError:
            print 'tried to index %s at %d' % (elmt, i)
            raise

    def indexed_func(i):
        def thunk():
            store_func_result()
            return i_th(cache['value'], i)
        return thunk

    return indexed_func


class CompositeReduction(Reduction):
    """
    A :class:`Reduction` made from combining :class:`Reduction`s.
    Results consist of lists of result from the component
    :class:`Reduction`s but the recursion is only performed once.

    NOTE: all of the reduce_xxx methods here return lists of length r,
    where r is the length of self.reductions.  All the get_reduced_xxx
    functions passed as arguments return lists of length s, where s is
    the number of subcomponents.

    The indices passed to the get_reduced_xxx_indexed() functions are
    0 <= i <= r.
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


def composite_reduction_type(dicts):
    KEYS = ['archive', 'bundle', 'collection', 'product',
            'fits_file', 'hdu', 'header_unit', 'data_unit']
    res = {}
    for k in KEYS:
        vs = [d[k] for d in dicts]
        res[k] = '(' + ', '.join(vs) + ')'
    return res
