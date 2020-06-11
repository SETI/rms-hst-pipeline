import os
import os.path
import shutil
import tarfile

import fs.path
from fs.base import FS
from fs.osfs import OSFS
import fs.walk

from pdart.archive.ChecksumManifest import (
    make_checksum_manifest,
    plain_lidvid_to_visits_dirpath,
)
from pdart.archive.TransferManifest import make_transfer_manifest
from pdart.db.BundleDB import _BUNDLE_DB_NAME, create_bundle_db_from_os_filepath
from pdart.fs.deliverablefs.DeliverableFS import DeliverableFS, lidvid_to_dirpath
from pdart.pds4.LIDVID import LIDVID
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import make_osfs, make_version_view

_TAR_NEEDED: bool = False


def _fix_up_deliverable(dir: str) -> None:
    # TODO DeliverableFS was written with an older directory
    # structure.  When used with the new, we get trailing dollar signs
    # on directories representing bundles, collections, and products.
    # No time to fix it right now, so we just patch up the resulting
    # directory tree.  TODO But *do* fix it.
    for path, _, _ in os.walk(dir, topdown=False):
        if path[-1] == "$":
            os.rename(path, path[:-1])


def copy_fs(version_view: FS, deliverable: FS) -> None:
    # TODO I could (and used to) just do a fs.copy.copy_fs() from the
    # version_view to a DeliverableFS.  I removed it to debug issues
    # with the validation tool.  Now I find this hack is just as easy
    # (though I wonder about efficiency).  It bothers me that this
    # visits hack parallels a visits hack in
    # plain_lidvid_to_visits_dirpath().  I should figure this out and
    # make it clean.  For now, though, this works.

    # Uses dollar-terminated paths
    for path, dirs, files in version_view.walk():
        parts = fs.path.parts(path)
        if len(parts) == 4:
            if len(parts[3]) == 10:
                visit = "visit_" + parts[3][4:6].lower() + "$"
                parts[3] = visit
        new_path = fs.path.join(*parts)
        if not deliverable.isdir(new_path):
            deliverable.makedir(new_path)
        for file in files:
            old_filepath = fs.path.join(path, file.name)
            new_filepath = fs.path.join(new_path, file.name)
            fs.copy.copy_file(version_view, old_filepath, deliverable, new_filepath)


class MakeDeliverable(MarkedStage):
    def _run(self) -> None:
        working_dir: str = self.working_dir()
        archive_dir: str = self.archive_dir()
        deliverable_dir: str = self.deliverable_dir()
        manifest_dir: str = self.manifest_dir()

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view:
            bundle_segment = self._bundle_segment

            os.mkdir(deliverable_dir)
            deliverable_osfs = OSFS(deliverable_dir)
            copy_fs(version_view, deliverable_osfs)
            _fix_up_deliverable(deliverable_dir)

            # open the database
            db_filepath = fs.path.join(working_dir, _BUNDLE_DB_NAME)
            db = create_bundle_db_from_os_filepath(db_filepath)

            # add manifests
            checksum_manifest_path = fs.path.join(manifest_dir, "checksum.manifest.txt")
            with open(checksum_manifest_path, "w") as f:
                f.write(make_checksum_manifest(db, plain_lidvid_to_visits_dirpath))

            transfer_manifest_path = fs.path.join(manifest_dir, "transfer.manifest.txt")
            with open(transfer_manifest_path, "w") as f:
                f.write(make_transfer_manifest(db, plain_lidvid_to_visits_dirpath))

            # Tar it up.
            if _TAR_NEEDED:
                bundle_dir = str(fs.path.join(deliverable_dir, self._bundle_segment))

                with tarfile.open(f"{bundle_dir}.tar", "w") as tar:
                    tar.add(bundle_dir, arcname=os.path.basename(bundle_dir))

                shutil.rmtree(bundle_dir)
