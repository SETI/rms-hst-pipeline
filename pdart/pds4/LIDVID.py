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

    def __eq__(self, other):
        return (self.lid == other.lid) and (self.vid == other.vid)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.lidvid

    def __repr__(self):
        return 'LIDVID(%r)' % self.lidvid
