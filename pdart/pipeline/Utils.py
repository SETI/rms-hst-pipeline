import os.path

# import sys
# import logging
# import pdslogger
from contextlib import contextmanager
from typing import Generator, Set

import fs.path

from fs.osfs import OSFS
from fs.wrap import read_only

from pdart.fs.cowfs.COWFS import COWFS
from pdart.fs.multiversioned.Multiversioned import Multiversioned
from pdart.fs.multiversioned.VersionView import VersionView
from pdart.fs.versioned.Versioned import (
    MultiversionedCOWFS,
    MultiversionedOSFS,
    SingleVersionedCOWFS,
    SingleVersionedOSFS,
)
from pdart.labels.CitationInformation import Citation_Information
from pdart.pds4.LID import LID
from pdart.pipeline.FSTypes import *


def show_tree(tag: str, fs: FS) -> None:
    line = f"---- {tag} "
    tag_len = len(tag)
    line += (60 - tag_len) * "-"
    print(line)
    fs.tree()


@contextmanager
def make_osfs(dir: str) -> Generator[OSFS, None, None]:
    """
    Create an OSFS filesystem with the directory as root.  Intended to
    be used in a with-statement.
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fs = OSFS(dir)
    yield fs
    fs.close()


@contextmanager
def make_mv_osfs(dir: str) -> Generator[OSFS, None, None]:
    """
    Create a multiversioned OSFS filesystem with a suffixed version of
    the directory path as its root.  Intended to be used in a
    with-statement.
    """
    fs = MultiversionedOSFS.create_suffixed(dir)
    yield fs
    fs.close()


@contextmanager
def make_sv_osfs(dir: str) -> Generator[OSFS, None, None]:
    """
    Create a single-versioned OSFS filesystem with a suffixed version
    of the directory path as its root.  Intended to be used in a
    with-statement.
    """
    fs = SingleVersionedOSFS.create_suffixed(dir)
    yield fs
    fs.close()


@contextmanager
def make_sv_deltas(base_fs: FS, cow_dirpath: str) -> Generator[COWFS, None, None]:
    """
    Create a single-versioned copy-on-write filesystem using the
    base_fs as its original contents, and with a suffixed version of
    the COW directory path as the root for the changes (also called
    the delta).  Intended to be used in a with-statement.
    """
    fs = SingleVersionedCOWFS.create_cowfs_suffixed(
        read_only(base_fs), cow_dirpath, True
    )
    cat_base_fs = categorize_filesystem(base_fs)
    cat_fs = categorize_filesystem(fs)
    if (
        cat_base_fs != EMPTY_FS_TYPE
        and cat_fs != EMPTY_FS_TYPE
        and cat_fs != cat_base_fs
    ):
        print(f"{cat_base_fs}, {cat_fs}")
        show_tree("base_fs", base_fs)
        show_tree("fs", fs)
        assert False
    yield fs
    fs.close()


@contextmanager
def make_mv_deltas(base_fs: FS, cow_dirpath: str) -> Generator[COWFS, None, None]:
    """
    Create a multiversioned copy-on-write filesystem using the base_fs
    as its original contents, and with a suffixed version of the COW
    directory path as the root for the changes (also called the
    delta).  Intended to be used in a with-statement.
    """
    fs = MultiversionedCOWFS.create_cowfs_suffixed(base_fs, cow_dirpath, True)
    cat_base_fs = categorize_filesystem(base_fs)
    cat_fs = categorize_filesystem(fs)
    if (
        cat_base_fs != EMPTY_FS_TYPE
        and cat_fs != EMPTY_FS_TYPE
        and cat_fs != cat_base_fs
    ):
        print(f"{cat_base_fs}, {cat_fs}")
        show_tree("base_fs", base_fs)
        show_tree("fs", fs)
        assert False
    yield fs
    fs.close()


@contextmanager
def make_multiversioned(archive_osfs: FS) -> Generator[Multiversioned, None, None]:
    assert categorize_filesystem(archive_osfs) in [
        EMPTY_FS_TYPE,
        MULTIVERSIONED_FS_TYPE,
    ], categorize_filesystem(archive_osfs)
    yield Multiversioned(archive_osfs)


@contextmanager
def make_version_view(
    archive_osfs: OSFS, bundle_segment: str
) -> Generator[VersionView, None, None]:
    """
    Create a read-only view of the latest version of the bundle.
    Intended to be used in a with-statement.
    """
    with make_multiversioned(archive_osfs) as mv:
        lid = LID("urn:nasa:pds:" + str(bundle_segment))
        res = mv.create_version_view(lid)
        assert categorize_filesystem(res) in [
            EMPTY_FS_TYPE,
            SINGLE_VERSIONED_FS_TYPE,
        ], categorize_filesystem(res)

        yield res
        res.close()


def create_citation_info(
    sv_deltas: COWFS, document_dir: str, document_files: Set[str]
) -> Citation_Information:
    # We sort only to make '.apt' appear before '.pro' since the
    # algorithm for '.apt' is more reliable.
    for basename in sorted(document_files):
        _, ext = fs.path.splitext(basename)
        if ext.lower() in [".apt", ".pro"]:
            filepath = fs.path.join(document_dir, basename)
            os_filepath = sv_deltas.getsyspath(filepath)
            return Citation_Information.create_from_file(os_filepath)

    # If you got here, there was no '.apt' or '.pro' file and so we don't
    # know how to make Citation_Information.
    raise Exception(
        f"{document_dir} contains only {document_files}; "
        "can't make Citation_Information"
    )
