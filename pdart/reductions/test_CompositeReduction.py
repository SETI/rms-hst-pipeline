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

if False:
    def test_type_documentation():
        d = {'archive': 'None',
             'bundle': 'None',
             'collection': 'None',
             'product': 'ProductLabel',
             'fits_file': 'dict',
             'hdu': 'dict',
             'header_unit': 'None',
             'data_unit': 'None'}
        print reduction_type_documentation(composite_reduction_type([d, d]))
        assert False
