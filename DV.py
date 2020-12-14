import os.path

import fs.path


from pdart.archive.ChecksumManifest import make_checksum_manifest
from pdart.archive.TransferManifest import make_transfer_manifest
from pdart.db.BundleDB import (
    _BUNDLE_DB_NAME,
    create_bundle_db_from_os_filepath,
)
from pdart.fs.deliverableview.DeliverableView import DeliverableView
from pdart.fs.multiversioned.VersionView import VersionView
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.Utils import make_multiversioned, make_osfs, make_version_view


def lidvid_to_dirpath(lidvid: LIDVID) -> str:
    lid = lidvid.lid()
    parts = lid.parts()
    return fs.path.join("/", *parts)


def run() -> None:
    proposal_id = 9059
    _bundle_segment = "hst_09059"
    dirs = make_directories()

    archive_dir = dirs.archive_dir(proposal_id)
    working_dir = dirs.working_dir(proposal_id)

    db_filepath = os.path.join(working_dir, _BUNDLE_DB_NAME)
    db_exists = os.path.isfile(db_filepath)
    assert db_exists, f"Database doesn't exist at {db_filepath}"
    bundle_db = create_bundle_db_from_os_filepath(db_filepath)

    with make_osfs(archive_dir) as archive_osfs, make_multiversioned(
        archive_osfs
    ) as mv:
        bundle_lid = LID.create_from_parts([_bundle_segment])
        bundle_lidvid = mv.latest_lidvid(bundle_lid)
        assert bundle_lidvid is not None
        bundle_lidvid_str = str(bundle_lidvid)
        version_view = VersionView(mv, bundle_lidvid)

        synth_files = dict()
        cm = make_checksum_manifest(bundle_db, bundle_lidvid_str, lidvid_to_dirpath)
        synth_files["/checksum.manifest.txt"] = cm.encode("utf-8")
        tm = make_transfer_manifest(bundle_db, bundle_lidvid_str, lidvid_to_dirpath)
        synth_files["/transfer.manifest.txt"] = tm.encode("utf-8")

        dv = DeliverableView(version_view, synth_files)
        print("----------------")
        dv.tree()
        print("----------------")
        print(dv.gettext("/transfer.manifest.txt"))


if __name__ == "__main__":
    run()
