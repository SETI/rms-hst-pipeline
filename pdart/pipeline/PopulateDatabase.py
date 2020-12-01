import os.path
from typing import List

import fs.path

from pdart.db.BundleDB import (
    BundleDB,
    _BUNDLE_DB_NAME,
    create_bundle_db_from_os_filepath,
)
from pdart.pipeline.ChangesDict import (
    CHANGES_DICT_NAME,
    ChangesDict,
    read_changes_dict,
    write_changes_dict,
)
from pdart.db.FitsFileDB import populate_database_from_fits_file
from pdart.fs.cowfs.COWFS import COWFS
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view
from pdart.xml.Pds4Version import DISP_LIDVID, HST_LIDVID, PDS4_LIDVID


# TODO Cut-and-pasted from BuildLabels.  Refactor this.
def lid_to_dir(lid: LID) -> str:
    return fs.path.join(*[part + "$" for part in lid.parts()])


def _populate_schema_collection(db: BundleDB, bundle_lidvid: str) -> None:
    # TODO We're assuming here that there will only ever be one schema
    # collection.  I'm not sure that's true.
    lid = LIDVID(bundle_lidvid).lid().extend_lid("schema")
    new_lidvid = LIDVID.create_from_lid_and_vid(lid, VID("1.0"))
    collection_lidvid = str(new_lidvid)
    db.create_schema_collection(collection_lidvid, bundle_lidvid)

    # TODO Hardcoded here. Is this what we want to do?
    for lidvid in [DISP_LIDVID, HST_LIDVID, PDS4_LIDVID]:
        db.create_schema_product(lidvid)


def _populate_bundle(changes_dict: ChangesDict, db: BundleDB) -> LIDVID:
    for lid, (vid, changed) in changes_dict.items():
        if changed and lid.is_bundle_lid():
            lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
            db.create_bundle(str(lidvid))
            # there's only one, so return it
            return lidvid
    assert False, "No changed bundle LID in changes_dict"


def _populate_collections(changes_dict: ChangesDict, db: BundleDB) -> None:
    for lid, (vid, changed) in changes_dict.items():
        if lid.is_collection_lid():
            lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
            bundle_lidvid = changes_dict.parent_lidvid(lidvid)
            if changed:
                if lid.collection_id == "document":
                    db.create_document_collection(str(lidvid), str(bundle_lidvid))
                elif lid.collection_id == "schema":
                    # it's created separately
                    _populate_schema_collection(db, str(bundle_lidvid))
                else:
                    db.create_other_collection(str(lidvid), str(bundle_lidvid))
            else:
                if changes_dict.changed(bundle_lidvid.lid()):
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
                        db.create_document_file(sys_filepath, doc_file, str(lidvid))
                else:
                    db.create_fits_product(str(lidvid), str(collection_lidvid))

                    fits_files = [
                        fits_file
                        for fits_file in sv_deltas.listdir(product_path)
                        if fs.path.splitext(fits_file)[1].lower() == ".fits"
                    ]
                    for fits_file in fits_files:
                        fits_file_path = fs.path.join(product_path, fits_file)
                        fits_os_path = sv_deltas.getsyspath(fits_file_path)
                        populate_database_from_fits_file(db, fits_os_path, str(lidvid))
            else:
                if changes_dict.changed(collection_lidvid.lid()):
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
        schema_collection_lid = LID.create_from_parts([self._bundle_segment, "schema"])
        if not changes_dict.contains(schema_collection_lid):
            changes_dict.set(schema_collection_lid, VID("1.0"), True)
            write_changes_dict(changes_dict, changes_path)

        db_filepath = os.path.join(working_dir, _BUNDLE_DB_NAME)
        db_exists = os.path.isfile(db_filepath)
        db = create_bundle_db_from_os_filepath(db_filepath)

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas:
            if not db_exists:
                db.create_tables()

            bundle_lidvid = _populate_bundle(changes_dict, db)
            _populate_collections(changes_dict, db)
            _populate_products(changes_dict, db, sv_deltas)

        assert db

        assert os.path.isfile(db_filepath), db_filepath
