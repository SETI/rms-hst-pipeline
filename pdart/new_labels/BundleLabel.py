"""Functionality to build a bundle label using a SQLite database."""

from pdart.new_labels.BundleLabelXml import *
from pdart.new_labels.CitationInformation import *
from pdart.new_labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_bundle_label(bundle_db, verify):
    # type: (BundleDB, bool) -> unicode
    """
    Create the label text for the bundle in the bundle database using
    the database connection.  If verify is True, verify the label
    against its XML and Schematron schemas.  Raise an exception if
    either fails.
    """
    bundle = bundle_db.get_bundle()
    bundle_lid = lidvid_to_lid(bundle.lidvid)  # Only used in placeholder
    proposal_id = bundle.proposal_id
    reduced_collections = [
        make_bundle_entry_member({'collection_lidvid': collection.lidvid})
        for collection
        in bundle_db.get_bundle_collections(bundle.lidvid)]

    label = make_label({
        'bundle_lid': lidvid_to_lid(bundle.lidvid),
        'bundle_vid': lidvid_to_vid(bundle.lidvid),
        'proposal_id': str(proposal_id),
        'Citation_Information': make_placeholder_citation_information(
            bundle_lid),
        'Bundle_Member_Entries': combine_nodes_into_fragment(
            reduced_collections)
    }).toxml()

    return pretty_and_verify(label, verify)
