from pdart.reductions.Reduction import *


def indexed(func):
    """
    Convert a no-argument function returning a list, into a function
    that, given an index i, returns the i-th elements of each element
    of the result of the original function.

    Note that the original function will either never be called (if
    the indexed function is never called), or will be called only once
    (if any indexed function is called).

    This is to make sure that unnecessary recursions are not
    performed.

    Because you can't write to a variable outside a function in
    Python, but you can read one, we make cache a dictionary
    and write to its value (which is not a variable).
    """
    cache = { 'set': False, 'value': None }

    def store_func_result():
        if not cache['set']:
            cache['set'] = True
            cache['value'] = func()

    def i_th(elmt, i):
        if elmt is None:
            return None
        else:
            return elmt[i]

    def indexed_func(i):
        store_func_result()
        return [i_th(elmt, i) for elmt in cache['value']]

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
                                    get_reduced_products(i))
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
        get_reduced_header_unit_indexed = indexed(get_reduced_header_unit)
        get_reduced_data_unit_indexed = indexed(get_reduced_data_unit)
        return [r.reduce_hdu(n, hdu,
                             get_reduced_header_unit_indexed(i),
                             get_reduced_data_unit_indexed(i))
                for i, r in enumerate(self.reductions)]

    def reduce_header_unit(self, n, get_header_unit):
        header_unit = get_header_unit()
        return [r.reduce_header_unit(n, header_unit) for r in self.reductions]

    def reduce_data_unit(self, n, get_data_unit):
        # TODO header_unit and data_unit shouldn't be functions.
        data_unit = get_data_unit()
        return [r.reduce_data_unit(n, data_unit) for r in self.reductions]
