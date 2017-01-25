"""
A one-item cache for database lookups.  Unclear how much it speeds
things up.
"""
from contextlib import closing

from pdart.pds4labels.DBCalls import get_bundle_info_db, get_collection_info_db

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import sqlite3
    from typing import Any
    from pdart.xml.Templates import TemplateDict


_last_bundle = None
# type: unicode
_last_bundle_result = None
# type: TemplateDict


def lookup_bundle(conn, bundle):
    # type: (sqlite3.Connection, unicode) -> TemplateDict
    """Perform a database lookup on the bundle using a one-item cache."""
    global _last_bundle, _last_bundle_result
    if (bundle != _last_bundle):
        with closing(conn.cursor()) as cursor:
            (label_filepath, proposal_id) = get_bundle_info_db(cursor, bundle)
        _last_bundle = bundle
        _last_bundle_result = {'label_filepath': label_filepath,
                               'proposal_id': proposal_id}
    return _last_bundle_result

_last_collection = None
# type: unicode
_last_collection_result = None
# type: TemplateDict


def lookup_collection(conn, collection):
    # type: (sqlite3.Connection, unicode) -> TemplateDict
    """
    Perform a database lookup on the collection using a one-item
    cache.
    """
    global _last_collection, _last_collection_result
    if (collection != _last_collection):
        with closing(conn.cursor()) as cursor:
            (bundle, instrument, inventory_filepath, inventory_name,
             label_filepath, prefix, suffix) = \
             get_collection_info_db(cursor, collection)
        _last_collection = collection
        _last_collection_result = {'bundle': bundle,
                                   'instrument': instrument,
                                   'inventory_filepath': inventory_filepath,
                                   'inventory_name': inventory_name,
                                   'label_filepath': label_filepath,
                                   'prefix': prefix,
                                   'suffix': suffix}
    return _last_collection_result
