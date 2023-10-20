"""Representation of a PDS4 VID."""
import functools
import re


@functools.total_ordering
class VID(object):
    """Representation of a PDS4 VID."""

    def __init__(self, vid_str: str) -> None:
        """
        Create a VID object from a string, raising an exception if
        the VID string is malformed.
        """
        vs = vid_str.split(".")

        # Check requirements
        if len(vid_str) > 255:
            raise ValueError("VID is too long.")
        if len(vs) != 2:
            raise ValueError(f"VID {vid_str} does not have two components.")
        for v in vs:
            if not re.match("\\A(0|[1-9][0-9]*)\\Z", v):
                raise ValueError(f"VID is non-numeric: {v}.")

        self._VID = vid_str
        self._major = int(vs[0])
        # the major version may not be zero
        if self._major == 0:
            raise ValueError("The major version cannot be zero.")
        self._minor = int(vs[1])

    def major(self) -> int:
        """Return the major version number."""
        return self._major

    def minor(self) -> int:
        """Return the minor version number."""
        return self._minor

    def next_major_vid(self) -> "VID":
        """Return the next major VID."""
        return VID(f"{self.major() + 1}.0")

    def next_minor_vid(self) -> "VID":
        """Return the next minor VID."""
        return VID(f"{self.major()}.{self.minor()+ 1}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VID):
            return NotImplemented
        return self.major() == other.major() and self.minor() == other.minor()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, VID):
            return NotImplemented
        if self.major() == other.major():
            return self.minor() < other.minor()
        else:
            return self.major() < other.major()

    def __hash__(self) -> int:
        return hash((self._major, self._minor))

    def __str__(self) -> str:
        return self._VID

    def __repr__(self) -> str:
        return f"VID({self._VID!r})"
