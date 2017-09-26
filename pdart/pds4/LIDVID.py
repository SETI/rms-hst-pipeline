"""Representation of a PDS4 LIDVID."""
from pdart.pds4.LID import LID
from pdart.pds4.VID import VID


class LIDVID(object):
    """Representation of a PDS4 LIDVID."""

    def __init__(self, str):
        # type: (str) -> None
        """
        Create a LIDVID object from a string, raising an exception if
        the LIDVID string is malformed.
        """
        segs = str.split('::')
        assert len(segs) == 2
        self._lidvid = str
        self._lid = LID(segs[0])
        self._vid = VID(segs[1])

    def lid(self):
        # type: () -> LID
        return self._lid

    def vid(self):
        # type: () -> VID
        return self._vid

    def next_major_lidvid(self):
        """Return the next major LIDVID."""
        # type: () -> LIDVID
        return LIDVID('%s::%s' % (self.lid(), self.vid().next_major_vid()))

    def next_minor_lidvid(self):
        """Return the next minor LIDVID."""
        # type: () -> LIDVID
        return LIDVID('%s::%s' % (self.lid(), self.vid().next_minor_vid()))

    def __cmp__(self, other):
        res = cmp(self.lid(), other.lid())
        if res == 0:
            res = cmp(self.vid(), other.vid())
        return res

    def __str__(self):
        return self._lidvid

    def __repr__(self):
        return 'LIDVID(%r)' % self._lidvid
