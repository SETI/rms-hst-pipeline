from fs.osfs import OSFS
from pdart.fs.DeliverableFS import DeliverableFS
import fs.path
import os

from pdart.pipeline.Utils import make_osfs, make_version_view, make_sv_deltas


def _fix_up_deliverable(dir):
    # type: (unicode) -> None

    # TODO DeliverableFS was written with an older directory
    # structure.  When used with the new, we get trailing dollar signs
    # on directories representing bundles, collections, and products.
    # No time to fix it right now, so we just patch up the resulting
    # directory tree.  TODO But *do* fix it.
    for path, _, _ in os.walk(dir, topdown=False):
        if path[-1] == "$":
            print "****", path, "=>", path[:-1], "****"
            os.rename(path, path[:-1])


def make_deliverable(bundle_segment, archive_dir, deliverable_dir):
    # type: (str, unicode, unicode) -> None
    with make_osfs(archive_dir) as archive_osfs, make_version_view(
        archive_osfs, bundle_segment
    ) as version_view:
        # Hack-ish: just trying to get everything into place
        os.mkdir(deliverable_dir)
        deliverable_osfs = OSFS(deliverable_dir)
        deliverable_fs = DeliverableFS(deliverable_osfs)
        fs.copy.copy_fs(version_view, deliverable_fs)

        _fix_up_deliverable(deliverable_dir)
