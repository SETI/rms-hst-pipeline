"""
Placeholder functions.
"""
import sys


def placeholder(product_id, tag):
    # type: (str, str) -> str
    """Return placeholder text for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return "### placeholder for %s ###" % tag


def placeholder_int(product_id, tag):
    # type: (str, str) -> str
    """Return a placeholder integer for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return "0"


def placeholder_time(product_id, tag):
    # type: (str, str) -> str
    """Return a placeholder time for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return "2000-01-02Z"


def placeholder_year(product_id, tag):
    # type: (str, str) -> str
    """Return a placeholder year for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return "2000"


def placeholder_float(product_id, tag):
    # type: (str, str) -> str
    """Return a placeholder float for an XML tag, noting the problem."""
    note_problem(product_id, tag)
    return "0.0"


def note_problem(product_id, tag):
    # type: (str, str) -> None
    """Note use of a placeholder function."""
    if False:
        print(("PROBLEM %s: %s" % (tag, product_id)))
        sys.stdout.flush()


def known_placeholder(product_id, tag):
    # type: (str, str) -> str
    """Return placeholder text for an XML tag."""
    return "### placeholder for %s ###" % tag
