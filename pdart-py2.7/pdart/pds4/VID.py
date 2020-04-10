"""Representation of a PDS4 VID."""
import re


class VID(object):
    """Representation of a PDS4 VID."""

    def __init__(self, str):
        # type: (str) -> None
        """
        Create a VID object from a string, raising an exception if
        the VID string is malformed.
        """
        vs = str.split('.')

        # Check requirements
        assert len(str) <= 255, 'VID is too long'
        assert len(vs) is 2, ('VID %s does not have two components' % str)
        for v in vs:
            assert re.match('\\A(0|[1-9][0-9]*)\\Z', v), \
                'VID is non-numeric: %s' % v

        self._VID = str
        self._major = int(vs[0])
        assert self._major != 0  # the major version may not be zero
        self._minor = int(vs[1])

    def major(self):
        # type: () -> int
        """Return the major version number."""
        return self._major

    def minor(self):
        # type: () -> int
        """Return the minor version number."""
        return self._minor

    def next_major_vid(self):
        # type: () -> VID
        """Return the next major VID."""
        return VID('%d.0' % (self.major() + 1))

    def next_minor_vid(self):
        # type: () -> VID
        """Return the next minor VID."""
        return VID('%d.%d' % (self.major(), self.minor() + 1))

    def __cmp__(self, other):
        if other is None:
            return 1
        res = self.major() - other.major()
        if res == 0:
            res = self.minor() - other.minor()
        return res

    def __hash__(self):
        return hash((self._major, self._minor))

    def __str__(self):
        return self._VID

    def __repr__(self):
        return 'VID(%r)' % self._VID
