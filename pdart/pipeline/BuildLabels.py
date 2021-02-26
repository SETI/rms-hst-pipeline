import logging
from typing import cast, Set

import fs.path
import os.path

from pdart.db.BundleDB import (
    BundleDB,
    _BUNDLE_DB_NAME,
    create_bundle_db_from_os_filepath,
)
from pdart.db.BundleWalk import BundleWalk
from pdart.db.SqlAlchTables import (
    BadFitsFile,
    BrowseFile,
    BrowseProduct,
    Bundle,
    Collection,
    DocumentCollection,
    DocumentProduct,
    FitsFile,
    OtherCollection,
)
from pdart.fs.cowfs.COWFS import COWFS
from pdart.fs.multiversioned.Utils import lid_to_dirpath
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
from pdart.labels.TargetIdentification import make_context_target_label
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.ChangesDict import (
    CHANGES_DICT_NAME,
    ChangesDict,
    read_changes_dict,
    write_changes_dict,
)
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import (
    make_osfs,
    make_sv_deltas,
    make_version_view,
)

import urllib
import bs4  # type: ignore

_LOGGER = logging.getLogger(__name__)


_VERIFY = False

PUBLICATION_YEAR = 2021
PDS_URL = "https://pds.nasa.gov/data/pds4/context-pds4/target/"


def log_label(tag: str, lidvid: str) -> None:
    _LOGGER.info(f"{tag} label for {lidvid}")


def _lidvid_to_dir(lidvid: str) -> str:
    def get_lid(lidvid: str) -> LID:
        return LIDVID(lidvid).lid()

    return lid_to_dirpath(get_lid(str(lidvid)))


def _extend_lidvid(lidvid_str: str, segment: str) -> str:
    lidvid = LIDVID(lidvid_str)
    lid = lidvid.lid().extend_lid(segment)
    new_lidvid = LIDVID.create_from_lid_and_vid(lid, lidvid.vid())
    return str(new_lidvid)


def create_pds4_labels(
    working_dir: str,
    bundle_db: BundleDB,
    bundle_lidvid: LIDVID,
    changes_dict: ChangesDict,
    label_deltas: COWFS,
    info: Citation_Information,
) -> None:
    class _CreateLabelsWalk(BundleWalk):
        def visit_bundle(self, bundle: Bundle, post: bool) -> None:
            if post:
                first_bundle = LIDVID(bundle.lidvid).vid() == VID("1.0")
                if first_bundle:
                    self._create_context_collection(bundle)
                    self._create_schema_collection(bundle)
                else:
                    context_collection_lid = (
                        LIDVID(bundle.lidvid).lid().extend_lid("context")
                    )
                    context_collection_lidvid = LIDVID.create_from_lid_and_vid(
                        context_collection_lid, VID("1.0")
                    )
                    bundle_db.create_context_collection(
                        str(context_collection_lidvid), bundle.lidvid
                    )
                    changes_dict.set(context_collection_lid, VID("1.0"), False)
                    bundle_db.create_bundle_collection_link(
                        str(bundle_lidvid), str(context_collection_lidvid)
                    )
                self._post_visit_bundle(bundle)

        def _create_context_collection(self, bundle: Bundle) -> None:
            context_products = bundle_db.get_context_products()
            if not context_products:
                return

            bundle_lidvid = str(bundle.lidvid)
            collection_lidvid = _extend_lidvid(bundle_lidvid, "context")
            bundle_db.create_context_collection(collection_lidvid, bundle_lidvid)
            clv = LIDVID(collection_lidvid)
            changes_dict.set(clv.lid(), clv.vid(), True)

            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            context_coll_dir_path = fs.path.join(bundle_dir_path, "context$")
            label_deltas.makedir(context_coll_dir_path)

            self._create_context_target_label(context_coll_dir_path, collection_lidvid)

            collection = bundle_db.get_collection(collection_lidvid)
            self._post_visit_collection(collection)

        def _create_context_target_label(
            self, context_coll_dir_path: str, collection_lidvid: str
        ) -> None:
            """
            Create target lable under context collection if it doesn't exist
            in PDS page.
            """
            with urllib.request.urlopen(PDS_URL) as response:
                html = response.read()
            soup = bs4.BeautifulSoup(html, "html.parser")
            a_tags = soup.find_all("a")
            target_label_list = [a.string for a in a_tags if a.string]
            target_records = bundle_db.get_all_target_identification()
            target_list = []
            for record in target_records:
                name = str(record.name).replace(" ", "_")
                target = f"{record.type}.{name}".lower()
                if target not in target_list:
                    target_list.append(target)

            for target in target_list:
                is_target_label_exists = False
                for label in target_label_list:
                    if target in label:
                        is_target_label_exists = True
                        break
                if not is_target_label_exists:
                    label_filename = f"{target}_1.0.xml"
                    label_filepath = fs.path.join(context_coll_dir_path, label_filename)
                    target_lidvid = f"urn:nasa:pds:context:target:{target}::1.0"
                    label = make_context_target_label(self.db, target, _VERIFY)
                    label_deltas.setbytes(label_filepath, label)
                    bundle_db.create_target_label(
                        label_deltas.getsyspath(label_filepath),
                        label_filename,
                        target_lidvid,
                        collection_lidvid,
                    )

        def _create_schema_collection(self, bundle: Bundle) -> None:
            schema_products = bundle_db.get_schema_products()
            if not schema_products:
                return

            bundle_lidvid = str(bundle.lidvid)
            collection_lidvid = _extend_lidvid(bundle_lidvid, "schema")
            bundle_db.create_schema_collection(collection_lidvid, bundle_lidvid)

            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            schema_coll_dir_path = fs.path.join(bundle_dir_path, "schema$")
            label_deltas.makedir(schema_coll_dir_path)
            collection = bundle_db.get_collection(collection_lidvid)
            self._post_visit_collection(collection)

        def _post_visit_bundle(self, bundle: Bundle) -> None:
            bundle_lidvid = str(bundle.lidvid)
            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            label = make_bundle_label(self.db, bundle_lidvid, info, _VERIFY)
            assert label[:6] == b"<?xml ", "Not XML"

            label_filename = "bundle.xml"
            label_filepath = fs.path.join(bundle_dir_path, label_filename)
            label_deltas.setbytes(label_filepath, label)
            log_label("bundle", bundle_lidvid)
            bundle_db.create_bundle_label(
                label_deltas.getsyspath(label_filepath), label_filename, bundle_lidvid
            )

        def _post_visit_collection(self, collection: Collection) -> None:
            """Common implementation for all collections."""
            if not changes_dict.changed(LIDVID(collection.lidvid).lid()):
                return
            collection_lidvid = collection.lidvid
            collection_dir_path = _lidvid_to_dir(collection_lidvid)

            inventory = make_collection_inventory(self.db, collection_lidvid)
            inventory_filename = get_collection_inventory_name(
                self.db, collection_lidvid
            )
            inventory_filepath = fs.path.join(collection_dir_path, inventory_filename)

            # TODO Remove this kludge to fix COWFS.setbytes() bug.
            if label_deltas.exists(inventory_filepath):
                label_deltas.remove(inventory_filepath)

            label_deltas.setbytes(inventory_filepath, inventory)
            bundle_db.create_collection_inventory(
                label_deltas.getsyspath(inventory_filepath),
                inventory_filename,
                collection_lidvid,
            )

            log_label("collection", collection_lidvid)
            label = make_collection_label(
                self.db, info, collection_lidvid, str(bundle_lidvid), _VERIFY
            )

            label_filename = get_collection_label_name(self.db, collection_lidvid)
            label_filepath = fs.path.join(collection_dir_path, label_filename)

            # TODO Remove this kludge to fix COWFS.setbytes() bug.
            if label_deltas.exists(label_filepath):
                label_deltas.remove(label_filepath)

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
                if not changes_dict.changed(LIDVID(document_collection.lidvid).lid()):
                    return
                self._post_visit_collection(document_collection)

        def visit_other_collection(
            self, bundle_lidvid: str, other_collection: OtherCollection, post: bool
        ) -> None:
            if post:
                if not changes_dict.changed(LIDVID(other_collection.lidvid).lid()):
                    return
                self._post_visit_collection(other_collection)

        def visit_document_product(
            self, collection_lidvid: str, document_product: DocumentProduct, post: bool
        ) -> None:
            if not post:
                return
            if not changes_dict.changed(LIDVID(document_product.lidvid).lid()):
                return
            product_lidvid = str(document_product.lidvid)

            # TODO publication date left blank
            publication_date = None
            log_label("document product", product_lidvid)
            label = make_document_product_label(
                self.db,
                info,
                product_lidvid,
                str(bundle_lidvid),
                _VERIFY,
                publication_date,
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
            # TODO Problem: browse products are not in the ChangesDict
            browse_product_lidvid = browse_file.product_lidvid
            browse_product = cast(
                BrowseProduct, bundle_db.get_product(browse_product_lidvid)
            )
            assert isinstance(browse_product, BrowseProduct)
            fits_product_lidvid = browse_product.fits_product_lidvid
            if not changes_dict.changed(LIDVID(fits_product_lidvid).lid()):
                return

            log_label("browse product", browse_file.product_lidvid)
            label = make_browse_product_label(
                self.db,
                collection_lidvid,
                str(browse_file.product_lidvid),
                str(browse_file.basename),
                str(bundle_lidvid),
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
            product_lidvid = bad_fits_file.product_lidvid
            if not changes_dict.changed(LIDVID(product_lidvid).lid()):
                return
            basename = bad_fits_file.basename
            assert False, (
                f"Not yet handling bad FITS file {basename} "
                f"in product {product_lidvid}"
            )

        def visit_fits_file(self, collection_lidvid: str, fits_file: FitsFile) -> None:
            if not changes_dict.changed(LIDVID(fits_file.product_lidvid).lid()):
                return
            log_label("FITS product", fits_file.product_lidvid)
            label = make_fits_product_label(
                working_dir,
                self.db,
                collection_lidvid,
                str(fits_file.product_lidvid),
                str(bundle_lidvid),
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

    _CreateLabelsWalk(bundle_db, str(bundle_lidvid)).walk()


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

        assert not os.path.isdir(
            self.deliverable_dir()
        ), "{deliverable_dir} cannot exist for BuildLabels"

        changes_path = fs.path.join(working_dir, CHANGES_DICT_NAME)
        changes_dict = read_changes_dict(changes_path)

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas, make_sv_deltas(
            sv_deltas, archive_browse_deltas_dir
        ) as browse_deltas, make_sv_deltas(
            browse_deltas, archive_label_deltas_dir
        ) as label_deltas:
            changes_path = fs.path.join(working_dir, CHANGES_DICT_NAME)
            changes_dict = read_changes_dict(changes_path)

            # open the database
            db_filepath = fs.path.join(working_dir, _BUNDLE_DB_NAME)
            db = create_bundle_db_from_os_filepath(db_filepath)

            # create labels
            bundle_lid = LID.create_from_parts([self._bundle_segment])
            bundle_vid = changes_dict.vid(bundle_lid)
            bundle_lidvid = LIDVID.create_from_lid_and_vid(bundle_lid, bundle_vid)

            documents_dir = f"/{self._bundle_segment}$/document$/phase2$"
            docs = set(sv_deltas.listdir(documents_dir))

            # fetch citation info from database
            citation_info_from_db = db.get_citation(str(bundle_lidvid))
            info = Citation_Information(
                citation_info_from_db.filename,
                citation_info_from_db.propno,
                citation_info_from_db.category,
                citation_info_from_db.cycle,
                citation_info_from_db.authors.split(","),
                citation_info_from_db.title,
                citation_info_from_db.submission_year,
                citation_info_from_db.timing_year,
            )
            info.set_publication_year(PUBLICATION_YEAR)

            # create_pds4_labels() may change changes_dict, because we
            # create the context collection if it doesn't exist.
            create_pds4_labels(
                working_dir, db, bundle_lidvid, changes_dict, label_deltas, info
            )
            write_changes_dict(changes_dict, changes_path)
