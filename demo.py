"""
This is a throw-away program to demonstrate built-in pyfilesystem2
filesystems, those written for PDART, and the Multiversioned object
that handles storing multiple bundles into a standard filesystem
directory.

It was written to be imported from a Python command line so you can
explore the filesystems at will.
"""

import os
import shutil
from typing import Optional, Tuple

# These imports are from pyfilesystem2
from fs.copy import *
from fs.osfs import *
from fs.memoryfs import *
from fs.tarfs import *

# These imports are for PDART.
from pdart.fs.cowfs.cowfs import *
from pdart.fs.deliverable_view.deliverable_view import *
from pdart.fs.multiversioned.multiversioned import *
from pdart.fs.multiversioned.version_view import *
from pdart.pds4.lid import LID
from pdart.pds4.lidvid import LIDVID


def reset() -> None:
    """
    Create a clean demo directory and, on Mac, open it in the Finder.
    """
    shutil.rmtree("demo", ignore_errors=True)
    os.mkdir("demo")
    os.system("open demo")


# Automatically create a clean demo directory each time you load this
# file.
reset()


def populate_fs(fsys: FS) -> None:
    """
    Put some stuff into a filesystem.  Note, this works for *any* kind
    of filesystem you can write to.
    """
    fsys.makedir("foo")
    with fsys.open("foo/bar.txt", "w") as f:
        f.write("Here is some text.\n")


def show_fs(fsys: FS, label: str = None) -> None:
    """
    Show the contents of a filesystem with some labeling.
    """
    if label:
        print("----", label, "----")
    fsys.tree()
    print("----------------")
    print()


def demo_basics() -> None:
    """
    Demostrate the functionality of three of the built-in
    pyfilesystems2 filesystems.
    """
    osfs = OSFS("demo")
    show_fs(osfs)
    populate_fs(osfs)
    show_fs(osfs, "in the OS's file system")
    osfs.close()

    memfs = MemoryFS()
    populate_fs(memfs)
    show_fs(memfs, "in memory")
    memfs.close()

    t = TarFS("demo/t.tar", write=True)
    populate_fs(t)
    show_fs(t, "in a TAR filesystem")
    t.close()


def set_up_cowfs() -> Tuple[FS, COWFS]:
    """
    Set up a copy-on-write filesystem with some data already in the
    read-only part.
    """
    os.makedirs("demo/read-only-area")
    os.makedirs("demo/read-write-area")
    ro = OSFS("demo/read-only-area")
    populate_fs(ro)
    rw = OSFS("demo/read-write-area")
    return ro, COWFS.create_cowfs(ro, rw)


def demo_cowfs() -> None:
    """
    Write into a copy-on-write filesystem and note that (1) the COW
    filesystem shows the changes but (2) the original read-only part
    is unaffected.
    """
    ro, c = set_up_cowfs()
    print(
        "contents of 'foo/bar.txt' in read-only fs:", repr(ro.readtext("foo/bar.txt"))
    )
    print(
        "contents of 'foo/bar.txt' in copy-on-write fs:",
        repr(c.readtext("foo/bar.txt")),
    )
    print("Now I'm going to overwrite 'foo/bar.txt'.")
    c.writetext("foo/bar.txt", "No, there's no text here.")
    print(
        "contents of 'foo/bar.txt' in read-only fs:", repr(ro.readtext("foo/bar.txt"))
    )
    print(
        "contents of 'foo/bar.txt' in copy-on-write fs:",
        repr(c.readtext("foo/bar.txt")),
    )
    print("The copy-on-write filesystem looks like this:")
    c.tree()
    print(
        "Now I'm going to delete 'foo/bar.txt'"
        " and add 'bar/', 'foo/baz.pdf', and 'foo/quux.txt'."
    )
    c.remove("foo/bar.txt")
    c.makedir("bar/")
    c.touch("foo/baz.pdf")
    c.touch("foo/quux.txt")
    print("Now the copy-on-write filesystem looks like this:")
    c.tree()
    print("But the original read-only filesystem still looks like this:")
    ro.tree()


# We define a global Multiversioned object so it can be used by more
# than demo function.
m: Optional[Multiversioned] = None
lid = LID("urn:nasa:pds:hst_00001")


def demo_multi() -> None:
    """
    Demonstration of the use of a Multiversioned object.
    """
    global m

    # Create a version 1 of a bundle in its own directory.
    os.mkdir("demo/v1")
    v1 = OSFS("demo/v1")
    v1.makedirs("hst_00001$/data_acs_raw$/j12345s$")
    v1.writetext(
        "hst_00001$/data_acs_raw$/j12345s$/j12345s_raw.fits", "This is a FITS file."
    )

    # Create the Multiversioned object.  It will use "demo/archive/"
    # for its storage.  Move version 1 of the bundle into it.
    os.mkdir("demo/archive")
    o = OSFS("demo/archive")
    m = Multiversioned(o)
    m.update_from_single_version(std_is_new, v1)

    print("The latest version of", lid, "is", m.latest_lidvid(lid))

    # Remove the storage for version 1 of the bundle, since it's now
    # in the archive.
    shutil.rmtree("demo/v1")

    # Create a version 2 of the bundle.  I've changed the contents of
    # the "FITS" file and added an XML label.
    os.mkdir("demo/v2")
    v2 = OSFS("demo/v2")
    v2.makedirs("hst_00001$/data_acs_raw$/j12345s$")
    v2.writetext(
        "hst_00001$/data_acs_raw$/j12345s$/j12345s_raw.fits",
        "Okay, I lied. This is not really a FITS file.",
    )
    v2.writetext(
        "hst_00001$/data_acs_raw$/j12345s$/j12345s_raw.xml",
        "Nor is this an PDS4 label.",
    )
    # Add version 2 into the archive.
    m.update_from_single_version(std_is_new, v2)

    print("The latest version of", lid, "is", m.latest_lidvid(lid))

    # Remove the storage for version 2 of the bundle, since it's now
    # in the archive.
    shutil.rmtree("demo/v2")


def demo_version_views() -> None:
    """
    Demonstration of the use of VersionViews to see individual
    versions within a Multiversioned object.
    """
    global m
    # Set up the Multiversioned archive.
    demo_multi()

    # Get a view on the first version and check the contents.
    vv1 = VersionView(m, LIDVID("urn:nasa:pds:hst_00001::1.0"))
    show_fs(vv1, "this is a view on version 1.0")
    print(
        "in version 1.0, j12345_raw.fits contains: ",
        vv1.readtext("hst_00001$/data_acs_raw$/j12345s$/j12345s_raw.fits"),
    )

    # Get a view on the second version and check *its* contents.
    vv2 = VersionView(m, LIDVID("urn:nasa:pds:hst_00001::2.0"))
    show_fs(vv2, "this is a view on version 2.0")
    print(
        "in version 2.0, j12345_raw.fits contains: ",
        vv2.readtext("hst_00001$/data_acs_raw$/j12345s$/j12345s_raw.fits"),
    )


def demo_deliverable() -> DeliverableView:
    """
    Demonstration of the DeliverableView: a view of one version of the
    bundle, in a human-friendly format.
    """
    global m

    # Create the DeliverableFS.
    os.makedirs("demo/hst_00001-deliverable")
    dos = OSFS("demo/hst_00001-deliverable")

    # Set up the Multiversioned archive.
    demo_multi()
    assert m

    # Get a view on one version.
    vv = m.create_version_view(lid)
    dv = DeliverableView(vv)

    show_fs(dv, "deliverable")

    copy_fs(dv, dos)
    return dv
