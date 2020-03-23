"""Representation of a PDS4 LIDVID."""

import functools

from pdart.pds4.LID import LID
from pdart.pds4.VID import VID


@functools.total_ordering
class LIDVID(object):
    """Representation of a PDS4 LIDVID."""

    def __init__(self, lidvid_str: str):
        """
        Create a LIDVID object from a string, raising an exception if
        the LIDVID string is malformed.
        """
        segs = lidvid_str.split("::")
        assert len(segs) == 2
        self._lidvid = lidvid_str
        self._lid = LID(segs[0])
        self._vid = VID(segs[1])

    @staticmethod
    def create_from_lid_and_vid(lid: LID, vid: VID) -> "LIDVID":
        return LIDVID("%s::%s" % (str(lid), str(vid)))

    def lid(self) -> LID:
        return self._lid

    def vid(self) -> VID:
        return self._vid

    def is_product_lidvid(self) -> bool:
        """Return True iff the LIDVID is a product LIDVID."""
        return self._lid.is_product_lid()

    def is_collection_lidvid(self) -> bool:
        """Return True iff the LIDVID is a collection LIDVID."""
        return self._lid.is_collection_lid()

    def is_bundle_lidvid(self) -> bool:
        """Return True iff the LIDVID is a bundle LIDVID."""
        return self._lid.is_bundle_lid()

    def next_major_lidvid(self) -> "LIDVID":
        """Return the next major LIDVID."""
        return LIDVID("%s::%s" % (self.lid(), self.vid().next_major_vid()))

    def next_minor_lidvid(self) -> "LIDVID":
        """Return the next minor LIDVID."""
        return LIDVID("%s::%s" % (self.lid(), self.vid().next_minor_vid()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LIDVID):
            raise NotImplemented
        return self.lid() == other.lid() and self.vid() == other.vid()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LIDVID):
            raise NotImplemented
        if self.lid() == other.lid():
            return self.vid() < other.vid()
        else:
            return self.lid() < other.lid()

    def __hash__(self) -> int:
        return hash((self._lid, self._vid))

    def __str__(self) -> str:
        return self._lidvid

    def __repr__(self) -> str:
        return "LIDVID(%r)" % self._lidvid
