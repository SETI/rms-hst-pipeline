from typing import List, cast

from fs.path import isabs
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID

from typing import Callable, Set, Union, Any
from fs.base import FS

_SUBCOMPS = Union[Set[LID], Set[LIDVID]]


def data_consistent(contains_lidvids: bool, subcomponents: _SUBCOMPS) -> bool:
    if contains_lidvids:
        return all([isinstance(subcomp, LIDVID) for subcomp in subcomponents])
    else:
        return all([isinstance(subcomp, LID) for subcomp in subcomponents])


class VersionContents(object):
    """
    The contents of a PDS4 versioned component (bundle, collection or
    product) as represented by a set of the LIDs or LIDVIDs of its
    subcomponents, and a list of file contents, represented by the FS
    in which they are stored and a set of filepaths to the files.

    Because it's useful to talk about the contents of either
    multiversion systems or single-version views, this data structure
    does double duty.  There is a flag to show that it contains
    LIDVIDs or not; this lets us distinguish the separate usages.
    """

    def __init__(
        self,
        contains_lidvids: bool,
        subcomponents: _SUBCOMPS,
        fs: FS,
        filepaths: Set[str],
    ) -> None:
        assert data_consistent(contains_lidvids, subcomponents)
        self.contains_lidvids = contains_lidvids
        self.subcomponents = subcomponents
        for filepath in filepaths:
            assert isabs(filepath)
            assert fs.isfile(filepath)
        self.fs = fs
        self.filepaths = filepaths

    def lidvids(self) -> List[LIDVID]:
        if self.contains_lidvids:
            return cast(List[LIDVID], list(self.subcomponents))
        else:
            raise TypeError(f"{self} contains LIDs, not LIDVIDs")

    def lids(self) -> List[LID]:
        if self.contains_lidvids:
            raise TypeError(f"{self} contains LIDVIDs, not LIDs")
        else:
            return cast(List[LID], list(self.subcomponents))

    def to_lid_version_contents(self) -> "VersionContents":
        if self.contains_lidvids:
            lids = {lidvid.lid() for lidvid in self.lidvids()}
            return VersionContents(False, lids, self.fs, self.filepaths)
        else:
            raise TypeError(f"{self} does not contain LIDVIDs")

    def filter_filepaths(self, filt: Callable[[str], bool]) -> "VersionContents":
        return VersionContents(
            self.contains_lidvids,
            self.subcomponents,
            self.fs,
            set(filter(filt, self.filepaths)),
        )

    ############################################################

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VersionContents):
            return False
        if self.contains_lidvids != other.contains_lidvids:
            raise TypeError("contains_lidvids for two VersionContents do not match")
        if self.subcomponents != other.subcomponents:
            return False
        if self.filepaths != other.filepaths:
            return False
        if self.fs == other.fs:
            return True
        for filepath in self.filepaths:
            # check each file
            if self.fs.readbytes(filepath) != other.fs.readbytes(filepath):
                return False
        return True

    def __str__(self):
        return (
            f"VersionContents({self.contains_lidvids}, "
            f"{self.subcomponents}, "
            f"{self.fs}, "
            f"{self.filepaths})"
        )

    def __repr__(self):
        return (
            f"VersionContents({self.contains_lidvids!r}, "
            f"{self.subcomponents!r}, "
            f"{self.fs!r}, "
            f"{self.filepaths!r})"
        )
