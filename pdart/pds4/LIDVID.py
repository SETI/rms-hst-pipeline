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

    @staticmethod
    def create_from_lid_and_vid(lid, vid):
        # type: (LID, VID) -> LIDVID
        return LIDVID('%s::%s' % (str(lid), str(vid)))

    def lid(self):
        # type: () -> LID
        return self._lid

    def vid(self):
        # type: () -> VID
        return self._vid

    def is_product_lidvid(self):
        # type: () -> bool
        """Return True iff the LIDVID is a product LIDVID."""
        return self._lid.is_product_lid()

    def is_collection_lidvid(self):
        # type: () -> bool
        """Return True iff the LIDVID is a collection LIDVID."""
        return self._lid.is_collection_lid()

    def is_bundle_lidvid(self):
        # type: () -> bool
        """Return True iff the LIDVID is a bundle LIDVID."""
        return self._lid.is_bundle_lid()

    def next_major_lidvid(self):
        # type: () -> LIDVID
        """Return the next major LIDVID."""
        return LIDVID('%s::%s' % (self.lid(), self.vid().next_major_vid()))

    def next_minor_lidvid(self):
        # type: () -> LIDVID
        """Return the next minor LIDVID."""
        return LIDVID('%s::%s' % (self.lid(), self.vid().next_minor_vid()))

    def __cmp__(self, other):
        res = cmp(self.lid(), other.lid())
        if res == 0:
            res = cmp(self.vid(), other.vid())
        return res

    def __hash__(self):
        return hash((self._lid, self._vid))

    def __str__(self):
        return self._lidvid

    def __repr__(self):
        return 'LIDVID(%r)' % self._lidvid
