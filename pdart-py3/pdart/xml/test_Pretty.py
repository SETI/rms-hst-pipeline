from pdart.xml.Pretty import pretty_print

_PP: bytes = b"""<?xml version="1.0"?>
<foobar>
  <baz/>
</foobar>
"""


def test_pretty_print() -> None:
    """Mostly a smoketest to force parsing of Pretty."""
    assert pretty_print(b"<foobar><baz></baz></foobar>") == _PP
