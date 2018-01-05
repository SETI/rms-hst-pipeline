from contextlib import closing
from datetime import date
import sys

from pdart.new_labels.DocumentProductLabelXml import *
from pdart.new_labels.Utils import lidvid_to_lid
from pdart.xml.Pretty import pretty_and_verify

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_document_product_label(bundle_db, document_product_lidvid, verify):
    # type: (BundleDB, str, bool) -> unicode
    """
    Create the label text for the document product in the bundle
    having this :class:`~pdart.pds4.LID` using the database
    connection.  If verify is True, verify the label against its XML
    and Schematron schemas.  Raise an exception if either fails.
    """
    # TODO Should we be using LIDVIDs instead of LIDs?
    bundle = bundle_db.get_bundle()
    bundle_lid = lidvid_to_lid(bundle.lidvid)
    proposal_id = bundle.proposal_id
    title = 'Summary of the observation plan for HST proposal %d' % proposal_id

    product_lid = lidvid_to_lid(document_product_lidvid)

    label = make_label({
            'bundle_lid': bundle_lid,
            'product_lid': product_lid,
            'title': title,
            'publication_date': date.today().isoformat(),
            'Citation_Information': make_citation_information(bundle_lid,
                                                              proposal_id),
            'Document_Edition': make_document_edition(
                '0.0',
                [('phase2.txt', '7-Bit ASCII Text')])
            }).toxml()
    return pretty_and_verify(label, verify)


#     product_lid = bundle_lid + ':document:phase2'
#     with closing(conn.cursor()) as cursor:
#         # get some unknown info
#         (label_fp, proposal_id) = get_document_product_info(cursor,
#                                                             product_lid)
#         pass
#
#     title = 'Summary of the observation plan for HST proposal %d' % \
# proposal_id
#
#     label = make_label({
#             'bundle_lid': bundle_lid,
#             'product_lid': product_lid,
#             'title': title,
#             'publication_date': date.today().isoformat(),
#             'Citation_Information': make_citation_information(bundle_lid,
#                                                               proposal_id),
#             'Document_Edition': make_document_edition(
#                 '0.0',
#                 [('phase2.txt', '7-Bit ASCII Text')])
#             }).toxml()
#     label = pretty_print(label)
#
#     with open(label_fp, 'w') as f:
#         f.write(label)
#
#     print 'product label for', product_lid
#     sys.stdout.flush()
#
#     if verify:
#         verify_label_or_raise(label)
#
#     return label
