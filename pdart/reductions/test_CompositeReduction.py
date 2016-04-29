from pdart.reductions.CompositeReduction import *


def test_transpose():
    """A sanity check to make sure transpose works."""
    l = [[1, 2], [3, 4], [5, 6]]
    expected = [[1, 3, 5], [2, 4, 6]]
    actual = transpose(l)
    assert expected == actual, 'Expected %s; got %s' % (expected, actual)


def test_indexed():
    # Test scenario: there are 3 subcomponents and 2 reductions in the
    # composite.  Indices for get_reduced_xxx_indexed() should be 0
    # and 1, and it should return lists of length 3.
    res = [[1, 2], [3, 4], [5, 6]]
    get_reduced_xxx_indexed = indexed(lambda: res)

    expected = [1, 3, 5]
    actual = get_reduced_xxx_indexed(0)
    assert expected == actual, 'Expected %s; got %s' % (expected, actual)

    expected = [2, 4, 6]
    actual = get_reduced_xxx_indexed(1)
    assert expected == actual, 'Expected %s; got %s' % (expected, actual)
