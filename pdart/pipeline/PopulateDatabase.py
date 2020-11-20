import os.path
from typing import List

import fs.path

from pdart.db.BundleDB import (
    BundleDB,
    _BUNDLE_DB_NAME,
    create_bundle_db_from_os_filepath,
)
from pdart.pipeline.ChangesDict import CHANGES_DICT_NAME, ChangesDict, read_changes_dict
from pdart.db.FitsFileDB import populate_database_from_fits_file
from pdart.fs.cowfs.COWFS import COWFS
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view
from pdart.xml.Pds4Version import DISP_LIDVID, HST_LIDVID, PDS4_LIDVID

_INITIAL_VID: VID = VID("1.0")


# TODO Cut-and-pasted from BuildLabels.  Refactor this.
def lid_to_dir(lid: LID) -> str:
    return fs.path.join(*[part + "$" for part in lid.parts()])


def _create_initial_lidvid_from_parts(parts: List[str]) -> str:
    lid = LID.create_from_parts(parts)
    lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(lidvid)


def _extend_initial_lidvid(lidvid: str, segment: str) -> str:
    lid = LIDVID(lidvid).lid().extend_lid(segment)
    new_lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(new_lidvid)


def _populate_schema_collection(db: BundleDB, bundle_lidvid: str) -> None:
    collection_lidvid = _extend_initial_lidvid(bundle_lidvid, "schema")
    db.create_schema_collection(collection_lidvid, bundle_lidvid)

    # TODO Hardcoded here. Is this what we want to do?
    for lidvid in [DISP_LIDVID, HST_LIDVID, PDS4_LIDVID]:
        db.create_schema_product(lidvid)


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


def _populate_from_other_collection(
    db: BundleDB,
    sv_deltas: COWFS,
    bundle_lidvid: str,
    collection_lidvid: str,
    collection_path: str,
) -> None:
    db.create_other_collection(collection_lidvid, bundle_lidvid)

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


def _populate_bundle(changes_dict: ChangesDict, db: BundleDB) -> None:
    for lid, (vid, changed) in changes_dict.items():
        if changed and lid.is_bundle_lid():
            lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
            print(f"%%%% db.create_bundle({lidvid})")
            db.create_bundle(str(lidvid))


def _populate_collections(changes_dict: ChangesDict, db: BundleDB) -> None:
    for lid, (vid, changed) in changes_dict.items():
        if lid.is_collection_lid():
            lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
            bundle_lidvid = changes_dict.parent_lidvid(lidvid)
            if changed:
                if lid.collection_id == "document":
                    print(
                        f"%%%% db.create_document_collection({str(lidvid)}, "
                        f"{str(bundle_lidvid)})"
                    )
                    db.create_document_collection(str(lidvid), str(bundle_lidvid))
                else:
                    print(
                        f"%%%% db.create_other_collection({str(lidvid)}, "
                        f"{str(bundle_lidvid)})"
                    )
                    db.create_other_collection(str(lidvid), str(bundle_lidvid))
            else:
                if changes_dict.changed(bundle_lidvid.lid()):
                    print(
                        f"%%%% db.create_bundle_collection_link({str(bundle_lidvid)}, "
                        f"{str(lidvid)})"
                    )
                    db.create_bundle_collection_link(str(bundle_lidvid), str(lidvid))


def _populate_products(
    changes_dict: ChangesDict, db: BundleDB, sv_deltas: COWFS
) -> None:
    for lid, (vid, changed) in changes_dict.items():
        if lid.is_product_lid():
            lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
            collection_lidvid = changes_dict.parent_lidvid(lidvid)
            if changed:
                product_path = lid_to_dir(lidvid.lid())
                if collection_lidvid.lid().collection_id == "document":
                    print(
                        f"%%%% db.create_document_product({str(lidvid)}, "
                        f"{str(collection_lidvid)})"
                    )
                    db.create_document_product(str(lidvid), str(collection_lidvid))

                    doc_files = [
                        doc_file
                        for doc_file in sv_deltas.listdir(product_path)
                        if (
                            fs.path.splitext(doc_file)[1].lower()
                            in [".apt", ".pdf", ".pro", ".prop"]
                        )
                    ]

                    for doc_file in doc_files:
                        sys_filepath = sv_deltas.getsyspath(
                            fs.path.join(product_path, doc_file)
                        )
                        print(
                            f"%%%% db.create_document_file({sys_filepath}, "
                            f"{doc_file}, "
                            f"{str(lidvid)})"
                        )
                        db.create_document_file(sys_filepath, doc_file, str(lidvid))
                else:
                    print(
                        f"%%%% db.create_fits_product({str(lidvid)}, "
                        f"{str(collection_lidvid)})"
                    )
                    db.create_fits_product(str(lidvid), str(collection_lidvid))

                    fits_files = [
                        fits_file
                        for fits_file in sv_deltas.listdir(product_path)
                        if fs.path.splitext(fits_file)[1].lower() == ".fits"
                    ]
                    for fits_file in fits_files:
                        fits_file_path = fs.path.join(product_path, fits_file)
                        fits_os_path = sv_deltas.getsyspath(fits_file_path)
                        print(
                            f"%%%% populate_database_from_fits_file(db, "
                            f"{fits_os_path}, "
                            f"{str(lidvid)})"
                        )
                        populate_database_from_fits_file(db, fits_os_path, str(lidvid))
            else:
                if changes_dict.changed(collection_lidvid.lid()):
                    print(
                        f"%%%% db.create_collection_product_link({str(collection_lidvid)}, "
                        f"{str(lidvid)})"
                    )
                    db.create_collection_product_link(
                        str(collection_lidvid), str(lidvid)
                    )


class PopulateDatabase(MarkedStage):
    """
    We create the database if necessary, create the bundle in the
    database, then walk through the new collections in the filesystem
    and create them in the database.

    When this stage finishes, there should be a database, populated
    with data from the primary files.
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        archive_dir: str = self.archive_dir()
        archive_primary_deltas_dir: str = self.archive_primary_deltas_dir()

        changes_path = os.path.join(working_dir, CHANGES_DICT_NAME)
        changes_dict = read_changes_dict(changes_path)

        db_filepath = os.path.join(working_dir, _BUNDLE_DB_NAME)
        db_exists = os.path.isfile(db_filepath)
        db = create_bundle_db_from_os_filepath(db_filepath)

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas:
            if db_exists:
                # second time through
                for lid, (vid, changed) in changes_dict.items():
                    print("####", lid, vid, changed)
                print("")
                _populate_bundle(changes_dict, db)
                _populate_collections(changes_dict, db)
                _populate_products(changes_dict, db, sv_deltas)
                assert False, "PopulateDatabase._run() not fully implemented"
            else:
                # first time through
                db.create_tables()
                bundle_lidvid = _create_initial_lidvid_from_parts(
                    [str(self._bundle_segment)]
                )
                db.create_bundle(bundle_lidvid)

                bundle_path = f"/{self._bundle_segment}$/"
                collection_segments = [
                    str(coll[:-1])
                    for coll in sv_deltas.listdir(bundle_path)
                    if "$" in coll
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
                            db,
                            sv_deltas,
                            bundle_lidvid,
                            collection_lidvid,
                            product_path,
                        )
                    else:
                        _populate_from_other_collection(
                            db,
                            sv_deltas,
                            bundle_lidvid,
                            collection_lidvid,
                            collection_path,
                        )
                _populate_schema_collection(db, bundle_lidvid)

        assert db

        assert os.path.isfile(db_filepath), db_filepath
