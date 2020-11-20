from typing import Set

import fs.path

from pdart.db.BundleDB import (
    BundleDB,
    _BUNDLE_DB_NAME,
    create_bundle_db_from_os_filepath,
)
from pdart.db.BundleWalk import BundleWalk
from pdart.db.SqlAlchTables import (
    BadFitsFile,
    BrowseFile,
    Bundle,
    Collection,
    DocumentCollection,
    DocumentProduct,
    FitsFile,
    OtherCollection,
)
from pdart.fs.cowfs.COWFS import COWFS
from pdart.labels.BrowseProductLabel import make_browse_product_label
from pdart.labels.BundleLabel import make_bundle_label
from pdart.labels.CitationInformation import Citation_Information
from pdart.labels.CollectionInventory import (
    get_collection_inventory_name,
    make_collection_inventory,
)
from pdart.labels.CollectionLabel import (
    get_collection_label_name,
    make_collection_label,
)
from pdart.labels.DocumentProductLabel import make_document_product_label
from pdart.labels.FitsProductLabel import make_fits_product_label
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view

_VERIFY = False


def _create_citation_info(
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


def _lidvid_to_dir(lidvid: str) -> str:
    def get_lid(lidvid: str) -> LID:
        return LIDVID(lidvid).lid()

    return lid_to_dir(get_lid(str(lidvid)))


def lid_to_dir(lid: LID) -> str:
    return fs.path.join(*[part + "$" for part in lid.parts()])


# TODO Cut-and-pasted from PopulateDatabase.  Refactor this.
_INITIAL_VID: VID = VID("1.0")


def _extend_initial_lidvid(lidvid: str, segment: str) -> str:
    lid = LIDVID(lidvid).lid().extend_lid(segment)
    new_lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(new_lidvid)


# END TODO


def create_pds4_labels(
    working_dir: str,
    bundle_db: BundleDB,
    label_deltas: COWFS,
    info: Citation_Information,
) -> None:
    class _CreateLabelsWalk(BundleWalk):
        def visit_bundle(self, bundle: Bundle, post: bool) -> None:
            if post:
                self._create_context_collection(bundle)
                self._create_schema_collection(bundle)
                self._post_visit_bundle(bundle)

        def _create_context_collection(self, bundle: Bundle) -> None:
            context_products = bundle_db.get_context_products()
            if not context_products:
                return

            bundle_lidvid = str(bundle.lidvid)
            collection_lidvid = _extend_initial_lidvid(bundle_lidvid, "context")
            bundle_db.create_context_collection(collection_lidvid, bundle_lidvid)

            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            context_coll_dir_path = fs.path.join(bundle_dir_path, "context$")
            label_deltas.makedir(context_coll_dir_path)
            collection = bundle_db.get_collection(collection_lidvid)
            self._post_visit_collection(collection)

        def _create_schema_collection(self, bundle: Bundle) -> None:
            schema_products = bundle_db.get_schema_products()
            if not schema_products:
                return

            bundle_lidvid = str(bundle.lidvid)
            collection_lidvid = _extend_initial_lidvid(bundle_lidvid, "schema")
            bundle_db.create_schema_collection(collection_lidvid, bundle_lidvid)

            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            schema_coll_dir_path = fs.path.join(bundle_dir_path, "schema$")
            label_deltas.makedir(schema_coll_dir_path)
            collection = bundle_db.get_collection(collection_lidvid)
            self._post_visit_collection(collection)

        def _post_visit_bundle(self, bundle: Bundle) -> None:
            bundle_lidvid = str(bundle.lidvid)
            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            label = make_bundle_label(self.db, info, _VERIFY)
            assert label[:6] == b"<?xml ", "Not XML"

            label_filename = "bundle.xml"
            label_filepath = fs.path.join(bundle_dir_path, label_filename)
            label_deltas.setbytes(label_filepath, label)
            bundle_db.create_bundle_label(
                label_deltas.getsyspath(label_filepath), label_filename, bundle_lidvid
            )

        def _post_visit_collection(self, collection: Collection) -> None:
            """Common implementation for all collections."""
            collection_lidvid = str(collection.lidvid)
            collection_dir_path = _lidvid_to_dir(collection_lidvid)

            inventory = make_collection_inventory(self.db, collection_lidvid)
            inventory_filename = get_collection_inventory_name(
                self.db, collection_lidvid
            )
            inventory_filepath = fs.path.join(collection_dir_path, inventory_filename)
            label_deltas.setbytes(inventory_filepath, inventory)
            bundle_db.create_collection_inventory(
                label_deltas.getsyspath(inventory_filepath),
                inventory_filename,
                collection_lidvid,
            )

            label = make_collection_label(self.db, info, collection_lidvid, _VERIFY)
            label_filename = get_collection_label_name(self.db, collection_lidvid)
            label_filepath = fs.path.join(collection_dir_path, label_filename)
            label_deltas.setbytes(label_filepath, label)
            bundle_db.create_collection_label(
                label_deltas.getsyspath(label_filepath),
                label_filename,
                collection_lidvid,
            )

        def visit_document_collection(
            self,
            bundle_lidvid: str,
            document_collection: DocumentCollection,
            post: bool,
        ) -> None:
            if post:
                self._post_visit_collection(document_collection)

        def visit_other_collection(
            self, bundle_lidvid: str, other_collection: OtherCollection, post: bool
        ) -> None:
            if post:
                self._post_visit_collection(other_collection)

        def visit_document_product(
            self, collection_lidvid: str, document_product: DocumentProduct, post: bool
        ) -> None:
            if not post:
                return
            product_lidvid = str(document_product.lidvid)

            # TODO publication date left blank
            publication_date = None
            label = make_document_product_label(
                self.db, info, product_lidvid, _VERIFY, publication_date
            )

            label_base = LIDVID(product_lidvid).lid().product_id
            assert label_base
            label_filename = label_base + ".xml"
            product_dir_path = _lidvid_to_dir(product_lidvid)
            label_filepath = fs.path.join(product_dir_path, label_filename)
            label_deltas.setbytes(label_filepath, label)
            bundle_db.create_product_label(
                label_deltas.getsyspath(label_filepath), label_filename, product_lidvid
            )

        def visit_browse_file(
            self, collection_lidvid: str, browse_file: BrowseFile
        ) -> None:
            label = make_browse_product_label(
                self.db,
                str(browse_file.product_lidvid),
                str(browse_file.basename),
                _VERIFY,
            )
            label_base = fs.path.splitext(browse_file.basename)[0]
            label_filename = label_base + ".xml"
            product_lidvid = str(browse_file.product_lidvid)
            product_dir_path = _lidvid_to_dir(product_lidvid)
            label_filepath = fs.path.join(product_dir_path, label_filename)
            label_deltas.setbytes(label_filepath, label)
            bundle_db.create_product_label(
                label_deltas.getsyspath(label_filepath), label_filename, product_lidvid
            )

        def visit_bad_fits_file(
            self, collection_lidvid: str, bad_fits_file: BadFitsFile
        ) -> None:
            basename = bad_fits_file.basename
            product_lidvid = bad_fits_file.product_lidvid
            assert False, (
                f"Not yet handling bad FITS file {basename} "
                f"in product {product_lidvid}"
            )

        def visit_fits_file(self, collection_lidvid: str, fits_file: FitsFile) -> None:
            label = make_fits_product_label(
                working_dir,
                self.db,
                str(fits_file.product_lidvid),
                str(fits_file.basename),
                _VERIFY,
            )
            label_base = fs.path.splitext(fits_file.basename)[0]
            label_filename = label_base + ".xml"
            product_lidvid = str(fits_file.product_lidvid)
            product_dir_path = _lidvid_to_dir(product_lidvid)
            label_filepath = fs.path.join(product_dir_path, label_filename)
            label_deltas.setbytes(label_filepath, label)
            bundle_db.create_product_label(
                label_deltas.getsyspath(label_filepath), label_filename, product_lidvid
            )

    _CreateLabelsWalk(bundle_db).walk()


class BuildLabels(MarkedStage):
    """
    Walk through the bundle in the database and create labels for the
    bundle, all collections and products, and collection inventories
    for all collections.
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        archive_dir: str = self.archive_dir()
        archive_primary_deltas_dir: str = self.archive_primary_deltas_dir()
        archive_browse_deltas_dir: str = self.archive_browse_deltas_dir()
        archive_label_deltas_dir: str = self.archive_label_deltas_dir()

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas, make_sv_deltas(
            sv_deltas, archive_browse_deltas_dir
        ) as browse_deltas, make_sv_deltas(
            browse_deltas, archive_label_deltas_dir
        ) as label_deltas:

            # open the database
            db_filepath = fs.path.join(working_dir, _BUNDLE_DB_NAME)
            db = create_bundle_db_from_os_filepath(db_filepath)

            # create labels
            bundle_lid = LID.create_from_parts([self._bundle_segment])
            bundle_lidvid = LIDVID.create_from_lid_and_vid(bundle_lid, VID("1.0"))
            documents_dir = f"/{self._bundle_segment}$/document$/phase2$"
            docs = set(sv_deltas.listdir(documents_dir))

            info = _create_citation_info(sv_deltas, documents_dir, docs)

            create_pds4_labels(working_dir, db, label_deltas, info)
