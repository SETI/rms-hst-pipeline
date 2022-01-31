import os
from typing import Dict

import fs.path
from fs.base import FS
from fs.osfs import OSFS

from pdart.archive.checksum_manifest import (
    make_checksum_manifest,
)
from pdart.archive.transfer_manifest import make_transfer_manifest
from pdart.db.bundle_db import _BUNDLE_DB_NAME, create_bundle_db_from_os_filepath
from pdart.fs.deliverable_view.deliverable_view import (
    DeliverableView,
    NO_VISIT_COLLECTIONS,
)
from pdart.fs.multiversioned.version_view import VersionView
from pdart.pds4.hst_filename import HstFilename
from pdart.pds4.lid import LID
from pdart.pds4.lidvid import LIDVID
from pdart.pipeline.ChangesDict import CHANGES_DICT_NAME, read_changes_dict
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.utils import make_multiversioned, make_osfs
from pdart.logging import PDS_LOGGER


def short_lidvid_to_dirpath(lidvid: LIDVID) -> str:
    lid = lidvid.lid()
    # parts are collection, product
    parts = lid.parts()[1:]
    if len(parts) >= 2 and parts[0] not in NO_VISIT_COLLECTIONS:
        fake_filename = f"{parts[1]}_raw.fits"
        visit = HstFilename(fake_filename).visit()
        visit_part = f"visit_{visit}"
        parts[1] = visit_part
    return fs.path.join(*parts)


def copy_fs(version_view: FS, deliverable: FS) -> None:
    # TODO The note below is probably obsolete.  It was written when
    # we were using the DeliverableFS to make the deliverable.  We've
    # now switched to using the DeliverableView. (The obsolete
    # DeliverableFS has been deleted from the repo.)  In any case,
    # this doesn't seem to hurt anything, but it can probably all be
    # replaced by fs.copy.copy_file()--see the last line.  Try it and
    # see.

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
    """
    Create a new directory with the bundle contents, organized in a
    more human-friendly hierarchy.  Create manifests for the
    deliverable and optionally tar it up into a tarball.
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        archive_dir: str = self.archive_dir()
        deliverable_dir: str = self.deliverable_dir()
        manifest_dir: str = self.manifest_dir()
        PDS_LOGGER.open("Create deliverable directory")
        if os.path.isdir(deliverable_dir):
            raise ValueError(f"{deliverable_dir} cannot exist for MakeDeliverable.")

        changes_path = os.path.join(working_dir, CHANGES_DICT_NAME)
        changes_dict = read_changes_dict(changes_path)

        with make_osfs(archive_dir) as archive_osfs, make_multiversioned(
            archive_osfs
        ) as mv:
            bundle_segment = self._bundle_segment
            bundle_lid = LID.create_from_parts([bundle_segment])
            bundle_vid = changes_dict.vid(bundle_lid)
            bundle_lidvid = LIDVID.create_from_lid_and_vid(bundle_lid, bundle_vid)
            version_view = VersionView(mv, bundle_lidvid)

            synth_files: Dict[str, bytes] = dict()

            # open the database
            db_filepath = fs.path.join(working_dir, _BUNDLE_DB_NAME)
            bundle_db = create_bundle_db_from_os_filepath(db_filepath)

            bundle_lidvid_str = str(bundle_lidvid)
            synth_files = dict()
            cm = make_checksum_manifest(
                bundle_db, bundle_lidvid_str, short_lidvid_to_dirpath
            )
            synth_files["/checksum.manifest.txt"] = cm.encode("utf-8")
            tm = make_transfer_manifest(
                bundle_db, bundle_lidvid_str, short_lidvid_to_dirpath
            )
            synth_files["/transfer.manifest.txt"] = tm.encode("utf-8")

            deliverable_view = DeliverableView(version_view, synth_files)

            os.mkdir(deliverable_dir)
            deliverable_osfs = OSFS(deliverable_dir)
            copy_fs(deliverable_view, deliverable_osfs)
            PDS_LOGGER.log("info", f"Deliverable: {deliverable_dir}")
        PDS_LOGGER.close()
