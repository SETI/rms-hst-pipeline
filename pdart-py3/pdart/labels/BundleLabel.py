"""Functionality to build a bundle label using a SQLite database."""

from pdart.citations import Citation_Information
from pdart.db.BundleDB import BundleDB
from pdart.labels.BundleLabelXml import make_bundle_entry_member, make_label
from pdart.labels.CitationInformation import make_citation_information
from pdart.labels.LabelError import LabelError
from pdart.labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify
from pdart.xml.Templates import combine_nodes_into_fragment


def make_bundle_label(
    bundle_db: BundleDB, info: Citation_Information, verify: bool
) -> bytes:
    """
    Create the label text for the bundle in the bundle database using
    the database connection.  If verify is True, verify the label
    against its XML and Schematron schemas.  Raise an exception if
    either fails.
    """
    bundle = bundle_db.get_bundle()
    proposal_id = bundle.proposal_id
    reduced_collections = [
        make_bundle_entry_member({"collection_lidvid": collection.lidvid})
        for collection in bundle_db.get_bundle_collections(bundle.lidvid)
    ]

    try:
        label = (
            make_label(
                {
                    "bundle_lid": lidvid_to_lid(bundle.lidvid),
                    "bundle_vid": lidvid_to_vid(bundle.lidvid),
                    "proposal_id": str(proposal_id),
                    "Citation_Information": make_citation_information(info),
                    "Bundle_Member_Entries": combine_nodes_into_fragment(
                        reduced_collections
                    ),
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(str(e), bundle.lidvid)

    assert label[:6] == b"<?xml ", "Not XML"
    return pretty_and_verify(label, verify)
