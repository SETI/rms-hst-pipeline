from abc import ABCMeta
import io
from typing import (
    Any,
    BinaryIO,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    cast,
)

from fs.base import FS
from fs.errors import ResourceReadOnly
from fs.info import Info
from fs.mode import check_writable
import fs.path
from fs.permissions import Permissions
from fs.subfs import SubFS

from pdart.db.bundle_db import BundleDB
from pdart.fs.multiversioned.multiversioned import Multiversioned
from pdart.fs.multiversioned.version_view import VersionView
from pdart.pds4.hst_filename import HstFilename
from pdart.pds4.lid import LID
from pdart.pds4.lidvid import LIDVID
from pdart.pds4.vid import VID

_INFO_DICT = Mapping[str, Mapping[str, object]]

NO_VISIT_COLLECTIONS = ["context", "document", "schema"]
# Do not convert paths within these collections to use "visit_NN"


def _first_index(pred: Callable[[str], bool], parts: List[str]) -> int:
    """
    Return the index to the first string in the list that satisfies
    the predicate.  Return the length of the list if no string does.
    """
    for i, part in enumerate(parts):
        if pred(part):
            return i
    return len(parts)


def translate_filepath(archive_filepath: str) -> str:
    """
    Translate an archive-style filepath (with "$" after bundle,
    collection and product directory names) into a deliverable-style
    filepath (with no "$"s and products organized into visit_NN
    directories instead of their own).
    """
    # The first part is always "/", so we drop it.
    parts = fs.path.parts(archive_filepath)[1:]
    # Special case for the root.
    if len(parts) == 0:
        return "/"

    # The path should consist of a sequence of parts all ending in
    # "$"--those show PDS4 structure--then the rest should *not* end
    # in "$": those make up the full path within the PDS4 bundle,
    # collection or product.  (Currently, we don't use subdirectories
    # within PDS4 objects, but it *is* allowed by the PDS4
    # specification.)
    indx = _first_index(lambda part: part[-1] != "$", parts)

    # Take the parts that end in "$", then drop the "$"s.
    pds4 = [part[:-1] for part in parts[:indx]]

    # Take the remaining parts that don't end in "$".
    non_pds4 = parts[indx:]

    if len(pds4) in [1, 2]:
        return fs.path.join("/", *pds4, *non_pds4)

    if len(pds4) == 3:
        collection_segment = pds4[1]
        if collection_segment in NO_VISIT_COLLECTIONS:
            return fs.path.join("/", *pds4, *non_pds4)
        else:
            product_segment = pds4[2]
            fake_filename = f"{product_segment}_raw.fits"
            visit = HstFilename(fake_filename).visit()
            return fs.path.join("/", *pds4[0:2], f"visit_{visit}", *non_pds4)

    raise ValueError(f"unexpected number of pds4 parts in {archive_filepath}")


class _Entry(object, metaclass=ABCMeta):
    """
    Entries in the DeliverableView's path dictionary.
    """

    pass


class _FileInfo(_Entry, metaclass=ABCMeta):
    """
    Entries that represent a file.
    """

    def __init__(self) -> None:
        _Entry.__init__(self)


class _BaseFileInfo(_FileInfo):
    """
    Entries that represent a file living in the base filesystem (the
    archive).
    """

    def __init__(self, archive_filepath: str) -> None:
        _FileInfo.__init__(self)
        self.archive_filepath = archive_filepath


class _SynthFileInfo(_FileInfo):
    """
    Entries that represent a synthetic file that doesn't live
    anywhere.  We use these to create the manifests on demand.
    """

    def __init__(self, contents: bytes) -> None:
        _FileInfo.__init__(self)
        self.contents = contents


class _DirInfo(_Entry):
    """
    Entries that represent a directory.  A directory may correspond to
    one in the base filesystem (like "/hst_12345"), but it also may be
    purely synthetic (like "/hst_12345/acs_data_raw/visit_89").
    """

    def __init__(self) -> None:
        _Entry.__init__(self)
        self.children: Dict[str, _Entry] = {}

    def add_dir(self, dirname: str) -> "_DirInfo":
        if dirname not in self.children:
            self.children[dirname] = _DirInfo()
        # This cast should be safe by construction.
        return cast(_DirInfo, self.children[dirname])

    def add_file(self, filename: str, file_info: _FileInfo) -> None:
        self.children[filename] = file_info


class DeliverableView(FS):
    """
    A human-friendly read-only view on a version of a bundle, possibly
    supplemented with some synthetic files that do not exist in the
    archive.  (We can use this facility to create manifest files on
    demand.)  Product files are collected into visit directories.

    Since the transfer manifest file encodes the layout of the
    deliverable, if we store it into the archive, we've hard-coded the
    layour permanently.  We're trying to avoid doing that.
    """

    def __init__(
        self, base_fs: VersionView, synth_files: Optional[Dict[str, bytes]] = None
    ) -> None:
        FS.__init__(self)
        self.base_fs = base_fs
        self.path_dict: Dict[str, _Entry] = {"/": _DirInfo()}

        self._populate_path_dict_from_base_fs()

        # Insert the synthetic files
        synth_files = dict() if synth_files is None else synth_files
        self._populate_path_dict_from_synth_files(synth_files)

    def _insert_dirpath(self, dirpath: str) -> _DirInfo:
        """
        Insert a directory path into the path dictionary and return
        its _DirInfo.  In doing this, we will create entries for any
        parent directories as necessary.
        """
        if dirpath == "/":
            # We've recursed to the top and just return the
            # _DirInfo value we initialized path_dict with.
            return cast(_DirInfo, self.path_dict["/"])
        else:
            parent_dirpath, dirname = fs.path.split(dirpath)
            # Make an entry for the parent directory
            # (recursively), and add your info to it.
            dir_info = self._insert_dirpath(parent_dirpath).add_dir(dirname)
            # Put your own entry into the path dictionary.
            self.path_dict[dirpath] = dir_info
            return dir_info

    def _populate_path_dict_from_synth_files(
        self, synth_files: Dict[str, bytes]
    ) -> None:
        def insert_synth_file(filepath: str, contents: bytes) -> None:
            """
            Insert a file path into the path dictionary and its
            contents.  In doing this, we will create entries for any
            parent directories as necessary.
            """
            dirpath, filename = fs.path.split(filepath)
            file_info = _SynthFileInfo(contents)
            # Make an entry for the parent directory (recursively),
            # and add your info to it.
            self._insert_dirpath(dirpath).add_file(filename, file_info)
            # Put your own entry into the path dictionary.
            self.path_dict[filepath] = file_info

        for filepath, contents in synth_files.items():
            insert_synth_file(filepath, contents)

    def _populate_path_dict_from_base_fs(self) -> None:
        """
        Recursively walk through all the filepaths for the VersionView
        and add them to the path dictionary.
        """

        def insert_filepaths(deliverable_filepath: str, archive_filepath: str) -> None:
            """
            Insert a file path into the path dictionary and the path
            to the file in the base filesystem.  In doing this, we
            will create entries for any parent directories as
            necessary.
            """
            file_info = _BaseFileInfo(archive_filepath)
            dirpath, filename = fs.path.split(deliverable_filepath)
            # Make an entry for the parent directory (recursively),
            # and add your info to it.
            self._insert_dirpath(dirpath).add_file(filename, file_info)
            # Put your own entry into the path dictionary.
            self.path_dict[deliverable_filepath] = file_info

        def insert_archive_filepath(archive_filepath: str) -> None:
            """
            Translate the filepath to the human-friendly deliverable
            format, and insert the two paths into the path dictionary.
            In doing this, we will create entries for any parent
            directories as necessary.
            """
            deliverable_filepath = translate_filepath(archive_filepath)
            insert_filepaths(deliverable_filepath, archive_filepath)

        for archive_filepath in self.base_fs.walk.files():
            insert_archive_filepath(archive_filepath)

    def getinfo(self, path: str, namespaces: Optional[Collection[str]] = None) -> Info:
        if path == "/":
            info = {"basic": {"name": "/", "is_dir": True}}
        else:
            _dirpath, name = fs.path.split(path)
            is_dir = isinstance(self.path_dict[path], _DirInfo)
            info = {"basic": {"name": name, "is_dir": is_dir}}
        return Info(info)

    def listdir(self, path: str) -> List[str]:
        entry = self.path_dict[path]
        if isinstance(entry, _DirInfo):
            return sorted(entry.children.keys())
        else:
            raise fs.errors.DirectoryExpected(path)

    def makedir(
        self,
        path: str,
        permissions: Optional[Permissions] = None,
        recreate: bool = False,
    ) -> SubFS["DeliverableView"]:
        self.check()
        raise ResourceReadOnly(path)

    def openbin(
        self, path: str, mode: str = "r", buffering: int = -1, **options: Any
    ) -> BinaryIO:
        self.check()
        if check_writable(mode):
            raise ResourceReadOnly(path)
        info = self.path_dict[fs.path.abspath(path)]
        if isinstance(info, _BaseFileInfo):
            return self.base_fs.openbin(
                info.archive_filepath, mode, buffering, **options
            )
        elif isinstance(info, _SynthFileInfo):
            return io.BytesIO(info.contents)
        else:
            raise fs.errors.FileExpected(path)

    def remove(self, path: str) -> None:
        self.check()
        raise ResourceReadOnly(path)

    def removedir(self, path: str) -> None:
        self.check()
        raise ResourceReadOnly(path)

    def setinfo(self, path: str, info: _INFO_DICT) -> None:
        self.check()
        raise ResourceReadOnly(path)

    @staticmethod
    def create_deliverable_view(
        bundle_db: BundleDB, mv: Multiversioned, lid: LID, vid: Optional[VID] = None
    ) -> "DeliverableView":
        if vid is None:
            vv = mv.create_version_view(lid)
        else:
            lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
            vv = VersionView(mv, lidvid)
        return DeliverableView(vv)
