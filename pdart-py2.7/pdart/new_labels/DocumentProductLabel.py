"""
Functionality to create a label for a document product.
"""
from datetime import date
from typing import TYPE_CHECKING

from pdart.new_labels.DocumentProductLabelXml \
    import make_doc_citation_information, make_document_edition, make_label
from pdart.new_labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from Citation_Information import Citation_Information
    from pdart.new_db.BundleDB import BundleDB


def make_document_product_label(bundle_db,
                                info,
                                document_product_lidvid,
                                verify,
                                publication_date=None):
    # type: (BundleDB, Citation_Information, str, bool, str) -> unicode
    """
    Create the label text for the document product in the bundle
    having this :class:`~pdart.pds4.LIDVID` using the database
    connection.  If verify is True, verify the label against its XML
    and Schematron schemas.  Raise an exception if either fails.
    """
    bundle = bundle_db.get_bundle()
    proposal_id = bundle.proposal_id
    title = 'Summary of the observation plan for HST proposal %d' % proposal_id

    product_lid = lidvid_to_lid(document_product_lidvid)
    product_vid = lidvid_to_vid(document_product_lidvid)
    publication_date = publication_date or date.today().isoformat()

    label = make_label({
        'bundle_lidvid': bundle.lidvid,
        'product_lid': product_lid,
        'product_vid': product_vid,
        'title': title,
        'publication_date': publication_date,
        'Citation_Information': make_doc_citation_information(info),

        # TODO don't the arguments to make_document_edition() need to not
        # be hard-coded?

        'Document_Edition': make_document_edition(
            '0.0',
            [('phase2.txt', '7-Bit ASCII Text')])
    }).toxml()
    return pretty_and_verify(label, verify)
