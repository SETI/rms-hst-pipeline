"""
Functionality to build a collection label using a SQLite database.
"""

from typing import TYPE_CHECKING

from pdart.new_labels.CitationInformation \
    import make_placeholder_citation_information
from pdart.new_labels.CollectionLabelXml import make_label
from pdart.new_labels.Utils import lidvid_to_lid
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_collection_inventory(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> unicode
    return u"I'm an inventory!"


def make_collection_label(bundle_db, collection_lidvid, verify):
    # type: (BundleDB, str, bool) -> unicode
    lid = lidvid_to_lid(collection_lidvid)
    collection = bundle_db.get_collection(collection_lidvid)
    suffix = collection.suffix
    proposal_id = bundle_db.get_bundle().proposal_id
    try:
        assert collection.prefix
        inventory_name = 'collection_%s.csv' % collection.prefix
    except:
        inventory_name = 'collection.csv'

    label = make_label({
        'lid': lid,
        'suffix': suffix,
        'proposal_id': str(proposal_id),
        'Citation_Information': make_placeholder_citation_information(lid),
        'inventory_name': inventory_name
    }).toxml()

    return pretty_and_verify(label, verify)

# def make_db_collection_inventory(conn, collection_lid):
#     # type: (sqlite3.Connection, unicode) -> unicode
#     """
#     Create the collection inventory for the collection having this
#     :class:`~pdart.pds4.LID` using the database connection and return
#     it.
#     """
#     with closing(conn.cursor()) as cursor:
#         lines = [u'P,%s\r\n' % str(product)
#                  for (product,)
#                  in get_collection_products_db(cursor, collection_lid)]
#     return ''.join(lines)
#
#
# def make_db_collection_label_and_inventory(conn, lid, verify):
#     # type: (sqlite3.Connection, unicode, bool) -> None
#     """
#     Create the label and inventory for the collection having this
#     :class:`~pdart.pds4.LID` using the database connection, writing
#     them to disk.  If verify is True, verify the label against its XML
#     and Schematron schemas.  Raise an exception if either fails.
#     """
#     d = lookup_collection(conn, lid)
#     label_fp = d['label_filepath']
#     bundle = d['bundle']
#     suffix = d['suffix']
#     inventory_name = d['inventory_name']
#     inventory_filepath = d['inventory_filepath']
#     proposal_id = lookup_bundle(conn, bundle)['proposal_id']
#
#     label = make_label({
#             'lid': lid,
#             'suffix': suffix,
#             'proposal_id': str(proposal_id),
#             'Citation_Information': make_placeholder_citation_information(
# lid),
#             'inventory_name': inventory_name
#     }).toxml()
#     label = pretty_print(label)
#
#     with open(label_fp, 'w') as f:
#         f.write(label)
#
#     if verify:
#         verify_label_or_raise(label)
#
#     with io.open(inventory_filepath, 'w', newline='') as f2:
#         inv = make_db_collection_inventory(conn, lid)
#         f2.write(unicode(inv))
#
#     print 'collection label and inventory for', lid
#     sys.stdout.flush()
#
