import LID
import VID


class LIDVID(object):
    """Representation of a PDS4 LIDVID."""

    def __init__(self, str):
        """
        Create a LIDVID object from a string, throwing an exception if
        the LIDVID string is malformed.
        """
        segs = str.split('::')
        assert len(segs) == 2
        self.lidvid = str
        self.lid = LID.LID(segs[0])
        self.vid = VID.VID(segs[1])

    def __cmp__(self, other):
        res = cmp(self.lid, other.lid)
        if res == 0:
            res = cmp(self.vid, other.vid)
        return res

    def __str__(self):
        return self.lidvid

    def __repr__(self):
        return 'LIDVID(%r)' % self.lidvid
