from typing import TYPE_CHECKING, List, cast

from fs.path import isabs
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Callable, Set, Union
    from fs.base import FS

    _SUBCOMPS = Union[Set[LID], Set[LIDVID]]


def data_consistent(contains_lidvids, subcomponents):
    # type: (bool, _SUBCOMPS) -> bool
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

    def __init__(self, contains_lidvids, subcomponents, fs, filepaths):
        # type: (bool, _SUBCOMPS, FS, Set[unicode]) -> None
        assert data_consistent(contains_lidvids, subcomponents)
        self.contains_lidvids = contains_lidvids
        self.subcomponents = subcomponents
        for filepath in filepaths:
            assert isabs(filepath)
            assert fs.isfile(filepath)
        self.fs = fs
        self.filepaths = filepaths

    def lidvids(self):
        # type: () -> List[LIDVID]
        if self.contains_lidvids:
            return cast(List[LIDVID], list(self.subcomponents))
        else:
            raise TypeError("%s contains LIDs, not LIDVIDs" % self)

    def lids(self):
        # type: () -> List[LID]
        if self.contains_lidvids:
            raise TypeError("%s contains LIDVIDs, not LIDs" % self)
        else:
            return cast(List[LID], list(self.subcomponents))

    def to_lid_version_contents(self):
        # type: () -> VersionContents
        if self.contains_lidvids:
            lids = {lidvid.lid() for lidvid in self.lidvids()}
            return VersionContents(False, lids, self.fs, self.filepaths)
        else:
            raise TypeError("%s does not contain LIDVIDs" % self)

    def filter_filepaths(self, filt):
        # type: (Callable[[unicode], bool]) -> VersionContents
        return VersionContents(
            self.contains_lidvids,
            self.subcomponents,
            self.fs,
            set(filter(filt, self.filepaths)),
        )

    ############################################################

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
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
        return "VersionContents(%s, %s, %s, %s)" % (
            self.contains_lidvids,
            self.subcomponents,
            self.fs,
            self.filepaths,
        )

    def __repr__(self):
        return "VersionContents(%r, %r, %r, %r)" % (
            self.contains_lidvids,
            self.subcomponents,
            self.fs,
            self.filepaths,
        )
