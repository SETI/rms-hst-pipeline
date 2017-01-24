"""
Placeholder functions.
"""
import sys


def placeholder(product_id, tag):
    # type: (unicode, unicode) -> unicode
    """Return placeholder text for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return '### placeholder for %s ###' % tag


def placeholder_int(product_id, tag):
    # type: (unicode, unicode) -> unicode
    """Return a placeholder integer for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return '0'


def placeholder_year(product_id, tag):
    # type: (unicode, unicode) -> unicode
    """Return a placeholder year for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return '2000'


def placeholder_float(product_id, tag):
    # type: (unicode, unicode) -> unicode
    """Return a placeholder float for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return '0.0'


def note_problem(product_id, tag):
    # type: (unicode, unicode) -> None
    """Note use of a placeholder function."""
    if False:
        print ('PROBLEM %s: %s' % (tag, product_id))
        sys.stdout.flush()


def known_placeholder(product_id, tag):
    # type: (unicode, unicode) -> unicode
    """Return placeholder text for an XML tag."""
    return '### placeholder for %s ###' % tag
