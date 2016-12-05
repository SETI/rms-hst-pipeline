"""Representation of a PDS4 VID."""
import re


class VID(object):
    """Representation of a PDS4 VID."""

    def __init__(self, str):
        # type: (unicode) -> None
        """
        Create a VID object from a string, raising an exception if
        the VID string is malformed.
        """
        vs = str.split('.')

        # Check requirements
        assert len(str) <= 255
        assert len(vs) == 2
        for v in vs:
            assert re.match('\\A(0|[1-9][0-9]*)\\Z', v)

        self.VID = str
        self.major = int(vs[0])
        self.minor = int(vs[1])

    def __cmp__(self, other):
        res = self.major - other.major
        if res == 0:
            res = self.minor - other.minor
        return res

    def __str__(self):
        return self.VID

    def __repr__(self):
        return 'VID(%r)' % self.VID
