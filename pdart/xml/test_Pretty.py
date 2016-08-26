from pdart.xml.Pretty import *

_PP = """<?xml version="1.0"?>
<foobar>
  <baz/>
</foobar>
"""


def test_pretty_print():
    """Mostly a smoketest to force parsing of Pretty."""
    assert pretty_print("<foobar><baz></baz></foobar>") == _PP
