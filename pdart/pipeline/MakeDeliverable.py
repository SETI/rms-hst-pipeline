import os
import os.path
import shutil
import subprocess
import tarfile

import fs.path
from fs.osfs import OSFS

from pdart.archive.ChecksumManifest import make_checksum_manifest
from pdart.archive.TransferManifest import make_transfer_manifest
from pdart.db.BundleDB import _BUNDLE_DB_NAME, create_bundle_db_from_os_filepath
from pdart.fs.primitives.DeliverableFS import DeliverableFS, lidvid_to_dirpath
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view


def _fix_up_deliverable(dir: str) -> None:
    # TODO DeliverableFS was written with an older directory
    # structure.  When used with the new, we get trailing dollar signs
    # on directories representing bundles, collections, and products.
    # No time to fix it right now, so we just patch up the resulting
    # directory tree.  TODO But *do* fix it.
    for path, _, _ in os.walk(dir, topdown=False):
        if path[-1] == "$":
            os.rename(path, path[:-1])


def make_deliverable(
    bundle_segment: str, working_dir: str, archive_dir: str, deliverable_dir: str
) -> None:
    with make_osfs(archive_dir) as archive_osfs, make_version_view(
        archive_osfs, bundle_segment
    ) as version_view:
        # Hack-ish: just trying to get everything into place
        os.mkdir(deliverable_dir)
        deliverable_osfs = OSFS(deliverable_dir)
        deliverable_fs = DeliverableFS(deliverable_osfs)
        fs.copy.copy_fs(version_view, deliverable_fs)

        _fix_up_deliverable(deliverable_dir)

        # open the database
        db_filepath = fs.path.join(working_dir, _BUNDLE_DB_NAME)
        db = create_bundle_db_from_os_filepath(db_filepath)

        # add manifests
        checksum_manifest_path = fs.path.join(deliverable_dir, "checksum.manifest.txt")
        with open(checksum_manifest_path, "w") as f:
            f.write(make_checksum_manifest(db, lidvid_to_dirpath))

        transfer_manifest_path = fs.path.join(deliverable_dir, "transfer.manifest.txt")
        with open(transfer_manifest_path, "w") as f:
            f.write(make_transfer_manifest(db, lidvid_to_dirpath))

        # Tar it up.
        TAR_NEEDED: bool = False
        if TAR_NEEDED:
            bundle_dir = str(fs.path.join(deliverable_dir, bundle_segment))

            with tarfile.open(f"{bundle_dir}.tar", "w") as tar:
                tar.add(bundle_dir, arcname=os.path.basename(bundle_dir))

            shutil.rmtree(bundle_dir)

        raise Exception("Succeeded but leaving a failure marker to prevent retries.")
