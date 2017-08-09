import os.path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple
    from pdart.pds4.Bundle import Bundle
    from pdart.pds4.Collection import Collection
    from pdart.pds4.Product import Product

    _CONN_TUPLE = \
        Tuple[str, str, str, str, unicode, unicode, unicode, unicode, unicode]
    _PROD_TUPLE = \
        Tuple[str, unicode, unicode, unicode, str, unicode, unicode]


BUNDLES_SCHEMA = """CREATE TABLE bundles (
        bundle VARCHAR PRIMARY KEY NOT NULL,
        full_filepath VARCHAR NOT NULL,
        label_filepath VARCHAR NOT NULL,
        proposal_id INT NOT NULL
        )"""
# type: str

BUNDLES_SQL = 'INSERT INTO bundles VALUES (?,?,?,?)'
# type: str


def bundle_tuple(bundle):
    # type: (Bundle) -> Tuple[str, unicode, unicode, int]
    return (str(bundle.lid),
            bundle.absolute_filepath(),
            bundle.label_filepath(),
            bundle.proposal_id())

COLLECTIONS_SCHEMA = """CREATE TABLE collections (
        collection VARCHAR PRIMARY KEY NOT NULL,
        full_filepath VARCHAR NOT NULL,
        label_filepath VARCHAR NOT NULL,
        bundle VARCHAR NOT NULL,
        prefix VARCHAR NOT NULL,
        suffix VARCHAR NOT NULL,
        instrument VARCHAR NOT NULL,
        inventory_name VARCHAR NOT NULL,
        inventory_filepath VARCHAR NOT NULL,
        FOREIGN KEY(bundle) REFERENCES bundles(bundle)
            );"""
# type: str

COLLECTIONS_SQL = 'INSERT INTO collections VALUES (?,?,?,?,?,?,?,?,?)'
# type: str


def collection_tuple(collection):
    # type: (Collection) -> _CONN_TUPLE
    return (str(collection.lid),
            collection.absolute_filepath(),
            collection.label_filepath(),
            str(collection.bundle().lid),
            collection.prefix(),
            collection.suffix(),
            collection.instrument(),
            collection.inventory_name(),
            collection.inventory_filepath())

PRODUCTS_SCHEMA = """CREATE TABLE products (
        product VARCHAR PRIMARY KEY NOT NULL,
        full_filepath VARCHAR NOT NULL,
        filename VARCHAR NOT NULL,
        label_filepath VARCHAR NOT NULL,
        collection VARCHAR NOT NULL,
        visit VARCHAR NOT NULL,
        hdu_count INT NOT NULL,
        product_id VARCHAR NOT NULL,
        FOREIGN KEY(collection) REFERENCES collections(collection)
        )"""
# type: str

PRODUCTS_SQL = 'INSERT INTO products VALUES (?,?,?,?,?,?,0,?)'
# type: str


def product_tuple(product):
    # type: (Product) -> _PROD_TUPLE
    return (str(product.lid),
            product.absolute_filepath(),
            os.path.basename(product.absolute_filepath()),
            product.label_filepath(),
            str(product.collection().lid),
            product.visit(),
            product.lid.product_id)


BAD_FITS_FILES_SCHEMA = """CREATE TABLE bad_fits_files (
        product VARCHAR NOT NULL,
        message VARCHAR NOT NULL,
        FOREIGN KEY (product) REFERENCES products(product)
        )"""
# type: str

BAD_FITS_FILES_SQL = 'INSERT INTO bad_fits_files VALUES (?,?)'
# type: str

HDUS_SCHEMA = """CREATE TABLE hdus (
        product VARCHAR NOT NULL,
        hdu_index INTEGER NOT NULL,
        hdrLoc INTEGER NOT NULL,
        datLoc INTEGER NOT NULL,
        datSpan INTEGER NOT NULL,
        FOREIGN KEY (product) REFERENCES products(product),
        CONSTRAINT hdus_pk PRIMARY KEY (product, hdu_index)
        )"""
# type: str

HDUS_SQL = 'INSERT INTO hdus VALUES (?, ?, ?, ?, ?)'
# type: str

CARDS_SCHEMA = """CREATE TABLE cards (
        keyword VARCHAR NOT NULL,
        value,
        product VARCHAR NOT NULL,
        hdu_index INTEGER NOT NULL,
        FOREIGN KEY(product) REFERENCES products(product),
        FOREIGN KEY(product, hdu_index) REFERENCES hdus(product, hdu_index)
        )"""
# type: str

CARDS_SQL = 'INSERT INTO cards VALUES (?, ?, ?, ?)'
# type: str

DOCUMENT_PRODUCTS_SCHEMA = """CREATE TABLE document_products (
    product VARCHAR PRIMARY KEY NOT NULL,
    label_filepath VARCHAR NOT NULL,
    proposal_id INTEGER NOT NULL,
    proposal_year INTEGER NOT NULL,
    pi_name VARCHAR NOT NULL,
    author_list VARCHAR NOT NULL,
    publication_year INTEGER NOT NULL)"""
# type: str

DOCUMENT_PRODUCTS_SQL = 'INSERT INTO document_products VALUES (?,?,?,?,?,?,?)'
# type: str

DOCUMENTS_SCHEMA = """CREATE TABLE documents (
    document_name VARCHAR PRIMARY KEY NOT NULL,
    document_filepath VARCHAR NOT NULL,
    source_url VARCHAR NOT NULL,
    product VARCHAR NOT NULL,
    FOREIGN KEY(product) REFERENCES document_products(product)"""
# type: str

DOCUMENTS_SQL = 'INSERT INTO documents VALUES (?,?,?,?)'
# type: str
