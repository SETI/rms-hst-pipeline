from pdart.reductions.Reduction import *


def indexed(func):
    """
    Convert a no-argument function returning a list, into a function
    that, given an index i, returns the i-th element of the result of
    the original function.

    Note that the original function will either never be called (if
    the indexed function is never called), or will be called only once
    (if any indexed function is called).

    This is to make sure that unnecessary recursions are not
    performed.

    Because you can't write to a variable outside a function in
    Python, but you can read one, we make cache a dictionary
    and write to its value (which is not a variable).

    (() -> [a1, a2, ..., an]) -> (i -> ai)
    """
    cache = {'set': False, 'value': None}

    def store_func_result():
        if not cache['set']:
            cache['set'] = True
            cache['value'] = func()

    def i_th(elmt, i):
        return elmt[i]

    def indexed_func(i):
        store_func_result()
        return cache['value'][i]

    return indexed_func

# For a composite of two reductions:
#
# archive reduces to (archive_1, archive_2)
# bundle reduces to (bundle_1, bundle_2)
# collection reduces to (collection_1, collection_2)
# product reduces to (product_1, product_2)
# fits_file reduces to (fits_file_1, fits_file_2)
# hdu reduces to (hdu_1, hdu_2)
# header_unit reduces to (header_unit_1, header_unit_2)
# data_unit reduces to (data_unit_1, data_unit_2)
#
# reduce_archive(
#     archive_root: str,
#     get_reduced_bundles: () -> [(bundle_1, bundle_2)])
#     ): (archive_1, archive_2)
#
# reduce_bundle(
#     archive: Archive,
#     lid: LID,
#     get_reduced_collections: () -> [(collection_1, collection_2)])
#     ): (bundle_1, bundle_2)
#
# reduce_collection(
#     archive: Archive,
#     lid: LID,
#     get_reduced_products: () -> [(product_1, product_2)])
#     ): (collection_1, collection_2)
#
# reduce_product(
#     archive: Archive,
#     lid: LID,
#     get_reduced_fits_files: () -> [(fits_file_1, fits_file_2)])
#     ): (product_1, product_2)
#
# reduce_fits_file(
#     file: string,
#     get_reduced_hdus: () -> [(hdu_1, hdu_2)])
#     ): (fits_file_1, fits_file_2)
#
# reduce_hdu(
#     n: int,
#     hdu: hdu,
#     get_reduced_header_unit: () -> (header_unit_1, header_unit_2),
#     get_reduced_data_unit: () -> (data_unit_1, data_unit_2))
#     : (hdu_1, hdu_2)
#
# reduce_header_unit(
#     n: int,
#     get_header_unit: () -> header_unit)
#     ): (header_unit_1, header_unit_2)
#
# reduce_data_unit(
#     n: int,
#     get_data_unit: () -> data_unit)
#     ): (data_unit_1, data_unit_2)


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


def composite_reduction_type(dicts):
    KEYS = ['archive', 'bundle', 'collection', 'product',
            'fits_file', 'hdu', 'header_unit', 'data_unit']
    res = {}
    for k in KEYS:
        vs = [d[k] for d in dicts]
        res[k] = '(' + ', '.join(vs) + ')'
    return res
