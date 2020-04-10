import fs.path
from typing import TYPE_CHECKING

from pdart.pipeline.Utils import make_osfs, make_version_view, make_sv_deltas
from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, create_bundle_db_from_os_filepath
from pdart.new_labels.DocumentProductLabel import make_document_product_label
from pdart.new_labels.FitsProductLabel import make_fits_product_label
from pdart.new_labels.BrowseProductLabel import make_browse_product_label
from pdart.new_db.BundleWalk import BundleWalk
from pdart.new_labels.BundleLabel import make_bundle_label
from pdart.new_labels.CollectionInventory import (
    get_collection_inventory_name,
    make_collection_inventory,
)
from pdart.new_labels.CollectionLabel import (
    get_collection_label_name,
    make_collection_label,
)
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.new_labels.CitationInformation import Citation_Information
from pdart.new_labels.Placeholders import placeholder

if TYPE_CHECKING:
    from typing import Set
    from pdart.fs.cowfs.COWFS import COWFS
    from pdart.new_db.BundleDB import BundleDB
    from pdart.new_db.SqlAlchTables import (
        BadFitsFile,
        BrowseFile,
        Bundle,
        Collection,
        DocumentCollection,
        DocumentProduct,
        FitsFile,
        FitsProduct,
        NonDocumentCollection,
    )

_VERIFY = True


def _placeholder_citation_information(proposal_id):
    # type: (int) -> Citation_Information
    bundle_id = "hst_%05d" % proposal_id
    return Citation_Information(
        placeholder(bundle_id, "filename"),
        placeholder(bundle_id, "propno"),
        placeholder(bundle_id, "category"),
        placeholder(bundle_id, "cycle"),
        placeholder(bundle_id, "authors"),
        placeholder(bundle_id, "title"),
        placeholder(bundle_id, "year"),
        placeholder(bundle_id, "author_list"),
        placeholder(bundle_id, "editor_list"),
        placeholder(bundle_id, "publication_year"),
        placeholder(bundle_id, "keyword"),
        placeholder(bundle_id, "description"),
    )


def _create_citation_info(sv_deltas, document_dir, document_files, proposal_id):
    # type: (COWFS, unicode, Set[unicode], int) -> Citation_Information

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
    return _placeholder_citation_information(proposal_id)


def _lidvid_to_dir(lidvid):
    # type: (str) -> unicode
    def get_lid(lidvid):
        # type: (str) -> LID
        return LIDVID(lidvid).lid()

    return lid_to_dir(get_lid(str(lidvid)))


def lid_to_dir(lid):
    # type: (LID) -> unicode
    return fs.path.join(*[part + "$" for part in lid.parts()])


def create_pds4_labels(bundle_db, label_deltas, info):
    # type: (BundleDB, COWFS, Citation_Information) -> None

    class _CreateLabelsWalk(BundleWalk):
        def visit_bundle(self, bundle, post):
            # type: (Bundle, bool) -> None
            if not post:
                return
            bundle_lidvid = str(bundle.lidvid)
            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            label = make_bundle_label(self.db, info, _VERIFY)
            label_filename = "bundle.xml"
            label_filepath = fs.path.join(bundle_dir_path, label_filename)
            label_deltas.settext(label_filepath, unicode(label))
            bundle_db.create_bundle_label(
                label_deltas.getsyspath(label_filepath), label_filename, bundle_lidvid
            )

        def _post_visit_collection(self, collection):
            # type: (Collection) -> None
            """Common implementation for all collections."""
            collection_lidvid = str(collection.lidvid)
            collection_dir_path = _lidvid_to_dir(collection_lidvid)

            inventory = make_collection_inventory(self.db, collection_lidvid)
            inventory_filename = get_collection_inventory_name(
                self.db, collection_lidvid
            )
            inventory_filepath = fs.path.join(collection_dir_path, inventory_filename)
            label_deltas.settext(inventory_filepath, unicode(inventory))
            bundle_db.create_collection_inventory(
                label_deltas.getsyspath(inventory_filepath),
                inventory_filename,
                collection_lidvid,
            )

            label = make_collection_label(self.db, info, collection_lidvid, _VERIFY)
            label_filename = get_collection_label_name(self.db, collection_lidvid)
            label_filepath = fs.path.join(collection_dir_path, label_filename)
            label_deltas.settext(label_filepath, unicode(label))
            bundle_db.create_collection_label(
                label_deltas.getsyspath(label_filepath),
                label_filename,
                collection_lidvid,
            )

        def visit_document_collection(self, document_collection, post):
            # type: (DocumentCollection, bool) -> None
            if post:
                self._post_visit_collection(document_collection)

        def visit_non_document_collection(self, non_document_collection, post):
            # type: (NonDocumentCollection, bool) -> None
            if post:
                self._post_visit_collection(non_document_collection)

        def visit_document_product(self, document_product, post):
            # type: (DocumentProduct, bool) -> None
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
            label_deltas.settext(label_filepath, unicode(label))
            bundle_db.create_product_label(
                label_deltas.getsyspath(label_filepath), label_filename, product_lidvid
            )

        def visit_browse_file(self, browse_file):
            # type: (BrowseFile) -> None
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
            label_deltas.settext(label_filepath, unicode(label))
            bundle_db.create_product_label(
                label_deltas.getsyspath(label_filepath), label_filename, product_lidvid
            )

        def visit_bad_fits_file(self, bad_fits_file):
            # type: (BadFitsFile) -> None
            assert False, "Not yet handling bad FITS file %s in product %s" % (
                str(bad_fits_file.basename),
                str(bad_fits_file.product_lidvid),
            )

        def visit_fits_file(self, fits_file):
            # type: (FitsFile) -> None
            label = make_fits_product_label(
                self.db, str(fits_file.product_lidvid), str(fits_file.basename), _VERIFY
            )
            label_base = fs.path.splitext(fits_file.basename)[0]
            label_filename = label_base + ".xml"
            product_lidvid = str(fits_file.product_lidvid)
            product_dir_path = _lidvid_to_dir(product_lidvid)
            label_filepath = fs.path.join(product_dir_path, label_filename)
            label_deltas.settext(label_filepath, unicode(label))
            bundle_db.create_product_label(
                label_deltas.getsyspath(label_filepath), label_filename, product_lidvid
            )

    _CreateLabelsWalk(bundle_db).walk()


def build_labels(
    proposal_id,
    bundle_segment,
    working_dir,
    archive_dir,
    archive_primary_deltas_dir,
    archive_browse_deltas_dir,
    archive_label_deltas_dir,
):
    # type: (int, str, unicode, unicode, unicode, unicode, unicode) -> None
    with make_osfs(archive_dir) as archive_osfs, make_version_view(
        archive_osfs, bundle_segment
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
        bundle_lid = LID.create_from_parts([bundle_segment])
        bundle_lidvid = LIDVID.create_from_lid_and_vid(bundle_lid, VID("1.0"))
        documents_dir = u"/%s$/document$/phase2$" % bundle_segment
        docs = set(sv_deltas.listdir(documents_dir))

        info = _create_citation_info(sv_deltas, documents_dir, docs, proposal_id)

        create_pds4_labels(db, label_deltas, info)
