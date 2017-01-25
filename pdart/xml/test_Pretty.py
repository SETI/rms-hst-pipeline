from pdart.xml.Pretty import *

_PP = """<?xml version="1.0"?>
<foobar>
  <baz/>
</foobar>
"""
# type: unicode


def test_pretty_print():
    # type: () -> None
    """Mostly a smoketest to force parsing of Pretty."""
    assert pretty_print("<foobar><baz></baz></foobar>") == _PP
