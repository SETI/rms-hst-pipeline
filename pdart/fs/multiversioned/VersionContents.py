from typing import Any, Callable, Generic, List, Set, TypeVar, Union, cast

from fs.base import FS
from fs.path import isabs
from fs.subfs import SubFS

from pdart.fs.multiversioned.Utils import component_files
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID


S = TypeVar("S", LID, LIDVID)


def _data_consistent(contains_lidvids: bool, subcomponents: Set[S]) -> bool:
    if contains_lidvids:
        return all([isinstance(subcomp, LIDVID) for subcomp in subcomponents])
    else:
        return all([isinstance(subcomp, LID) for subcomp in subcomponents])


class VersionContents(Generic[S]):
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
        subcomponents: Set[S],
        fs: FS,
        filepaths: Set[str],
    ) -> None:
        """
        Create a 'VersionContents' object.  DO NOT USE this
        constructor.  Instead use the static methods
        VersionContents.create_from_lids() and
        VersionContents.create_from_lidvids().
        """
        if not _data_consistent(contains_lidvids, subcomponents):
            raise TypeError(
                "Subcomponents are not consistent, should be all "
                + "livids or all lids."
            )
        self.contains_lidvids = contains_lidvids
        self.subcomponents: Set[S] = subcomponents
        for filepath in filepaths:
            if not isabs(filepath):
                raise ValueError(f"{filepath} is not an absolute path.")
            if not fs.isfile(filepath):
                raise ValueError(f"{filepath} is not a file.")
        self.fs = fs
        self.filepaths = filepaths

    @staticmethod
    def create_from_lids(
        subcomponents: Set[LID], fs: FS, filepaths: Set[str]
    ) -> "VersionContents[LID]":
        return VersionContents[LID](False, subcomponents, fs, filepaths)

    @staticmethod
    def create_from_lids_and_dirpath(
        subcomponents: Set[LID], fs: FS, dirpath: str
    ) -> "VersionContents[LID]":
        filepaths = component_files(fs, dirpath)
        return VersionContents[LID](
            False, subcomponents, SubFS(fs, dirpath), set(filepaths)
        )

    @staticmethod
    def create_from_lidvids(
        subcomponents: Set[LIDVID], fs: FS, filepaths: Set[str]
    ) -> "VersionContents[LIDVID]":
        return VersionContents[LIDVID](True, subcomponents, fs, filepaths)

    @staticmethod
    def create_from_lidvids_and_dirpath(
        subcomponents: Set[LIDVID], fs: FS, dirpath: str
    ) -> "VersionContents[LIDVID]":
        filepaths = component_files(fs, dirpath)
        return VersionContents[LIDVID](
            True, subcomponents, SubFS(fs, dirpath), set(filepaths)
        )

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

    def to_lid_version_contents(self) -> "VersionContents[LID]":
        if self.contains_lidvids:
            lids = {lidvid.lid() for lidvid in self.lidvids()}
            return VersionContents.create_from_lids(lids, self.fs, self.filepaths)
        else:
            raise TypeError(f"{self} does not contain LIDVIDs")

    def filter_filepaths(self, filt: Callable[[str], bool]) -> "VersionContents[S]":
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

    def __str__(self) -> str:
        return (
            f"VersionContents({self.contains_lidvids}, "
            f"{self.subcomponents}, "
            f"{self.fs}, "
            f"{self.filepaths})"
        )

    def __repr__(self) -> str:
        return (
            f"VersionContents({self.contains_lidvids!r}, "
            f"{self.subcomponents!r}, "
            f"{self.fs!r}, "
            f"{self.filepaths!r})"
        )
