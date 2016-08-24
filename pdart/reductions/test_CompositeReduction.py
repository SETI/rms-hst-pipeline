from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.reductions.CompositeReduction import *


def test_indexed():
    res = [[1, 2], [3, 4], [5, 6]]
    get_reduced_xxx_indexed = indexed(lambda: res)

    expected = [1, 3, 5]
    actual = get_reduced_xxx_indexed(0)()
    assert expected == actual, 'Expected %s; got %s' % (expected, actual)

    expected = [2, 4, 6]
    actual = get_reduced_xxx_indexed(1)()
    assert expected == actual, 'Expected %s; got %s' % (expected, actual)


class TracingReduction(Reduction):
    def __init__(self):
        self.trace = set()

    def reduce_archive(self, archive_root, get_reduced_bundles):
        self.trace.add('archive')

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        self.trace.add('bundle')

    def reduce_collection(self, archive, lid, get_reduced_products):
        self.trace.add('collection')

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        self.trace.add('product')

    def reduce_fits_file(self, file, get_reduced_hdus):
        self.trace.add('fits_file')

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        self.trace.add('hdu')

    def reduce_header_unit(self, n, header_unit):
        self.trace.add('header_unit')

    def reduce_data_unit(self, n, data_unit):
        self.trace.add('data_unit')


class TracingReduction2(TracingReduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        get_reduced_bundles()
        self.trace.add('archive')

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        get_reduced_collections()
        self.trace.add('bundle')


def test_composite_descent():
    arch = get_any_archive()
    tr = TracingReduction()
    cr = CompositeReduction([tr])
    # raise_verbosely(lambda: run_reduction(cr, arch))
    run_reduction(cr, arch)
    # Shouldn't recurse any deeper than archive-level.
    assert set(['archive']) == tr.trace

    tr = TracingReduction2()
    cr = CompositeReduction([tr])
    run_reduction(cr, arch)
    # Should recurse deeper.
    assert set(['archive', 'bundle', 'collection']) == tr.trace


# if False:
#     def test_type_documentation():
#         d = {'archive': 'None',
#              'bundle': 'None',
#              'collection': 'None',
#              'product': 'ProductLabel',
#              'fits_file': 'dict',
#              'hdu': 'dict',
#              'header_unit': 'None',
#              'data_unit': 'None'}
#         print reduction_type_documentation(composite_reduction_type([d, d]))
#         assert False
