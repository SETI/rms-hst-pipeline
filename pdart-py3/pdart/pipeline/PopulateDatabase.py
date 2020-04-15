import os.path
from typing import List

import fs.path

from pdart.db.BundleDB import (
    BundleDB,
    _BUNDLE_DB_NAME,
    create_bundle_db_from_os_filepath,
)
from pdart.db.FitsFileDB import populate_database_from_fits_file
from pdart.fs.cowfs.COWFS import COWFS
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Stage import Stage
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view

_INITIAL_VID: VID = VID("1.0")


def _create_initial_lidvid_from_parts(parts: List[str]) -> str:
    lid = LID.create_from_parts(parts)
    lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(lidvid)


def _extend_initial_lidvid(lidvid: str, segment: str) -> str:
    lid = LIDVID(lidvid).lid().extend_lid(segment)
    new_lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(new_lidvid)


def _populate_from_document_collection(
    db: BundleDB,
    sv_deltas: COWFS,
    bundle_lidvid: str,
    collection_lidvid: str,
    product_path: str,
) -> None:
    db.create_document_collection(collection_lidvid, bundle_lidvid)
    product_lidvid = _extend_initial_lidvid(collection_lidvid, "phase2")
    db.create_document_product(product_lidvid, collection_lidvid)
    for basename in sv_deltas.listdir(product_path):
        sys_filepath = sv_deltas.getsyspath(fs.path.join(product_path, basename))
        db.create_document_file(sys_filepath, basename, product_lidvid)


def _populate_from_non_document_collection(
    db: BundleDB,
    sv_deltas: COWFS,
    bundle_lidvid: str,
    collection_lidvid: str,
    collection_path: str,
) -> None:
    db.create_non_document_collection(collection_lidvid, bundle_lidvid)

    product_segments = [
        str(prod[:-1]) for prod in sv_deltas.listdir(collection_path) if "$" in prod
    ]
    for product_segment in product_segments:
        product_path = f"{collection_path}{product_segment}$/"
        product_lidvid = _extend_initial_lidvid(collection_lidvid, product_segment)
        fits_files = [
            fits_file
            for fits_file in sv_deltas.listdir(product_path)
            if fs.path.splitext(fits_file)[1] == ".fits"
        ]
        for fits_file in fits_files:
            fits_file_path = fs.path.join(product_path, fits_file)
            db.create_fits_product(product_lidvid, collection_lidvid)
            fits_os_path = sv_deltas.getsyspath(fits_file_path)

            populate_database_from_fits_file(db, fits_os_path, product_lidvid)


class PopulateDatabase(Stage):
    def _run(self) -> None:
        working_dir: str = self.dirs.working_dir(self.proposal_id)
        archive_dir: str = self.dirs.archive_dir(self.proposal_id)
        archive_primary_deltas_dir: str = self.dirs.archive_primary_deltas_dir(
            self.proposal_id
        )

        db_filepath = os.path.join(working_dir, _BUNDLE_DB_NAME)
        if os.path.isfile(db_filepath):
            db = create_bundle_db_from_os_filepath(db_filepath)
        else:
            db = create_bundle_db_from_os_filepath(db_filepath)
            db.create_tables()
            bundle_lidvid = _create_initial_lidvid_from_parts(
                [str(self.bundle_segment)]
            )
            db.create_bundle(bundle_lidvid)

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self.bundle_segment
        ) as version_view, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas:
            bundle_path = f"/{self.bundle_segment}$/"
            collection_segments = [
                str(coll[:-1]) for coll in sv_deltas.listdir(bundle_path) if "$" in coll
            ]
            for collection_segment in collection_segments:
                is_document_collection = collection_segment == "document"
                collection_path = f"{bundle_path}{collection_segment}$/"
                collection_lidvid = _extend_initial_lidvid(
                    bundle_lidvid, collection_segment
                )
                if is_document_collection:
                    product_path = f"{collection_path}phase2$/"
                    _populate_from_document_collection(
                        db, sv_deltas, bundle_lidvid, collection_lidvid, product_path
                    )
                else:
                    _populate_from_non_document_collection(
                        db, sv_deltas, bundle_lidvid, collection_lidvid, collection_path
                    )

        assert db

        assert os.path.isfile(db_filepath), db_filepath
