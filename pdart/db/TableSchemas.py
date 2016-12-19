BUNDLES_SCHEMA = """CREATE TABLE bundles (
        bundle VARCHAR PRIMARY KEY NOT NULL,
        full_filepath VARCHAR NOT NULL,
        label_filepath VARCHAR NOT NULL,
        proposal_id INT NOT NULL
        )"""
# type: str

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

BAD_FITS_FILES_SCHEMA = """CREATE TABLE bad_fits_files (
        product VARCHAR NOT NULL,
        message VARCHAR NOT NULL,
        FOREIGN KEY (product) REFERENCES products(product)
        )"""
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

CARDS_SCHEMA = """CREATE TABLE cards (
        keyword VARCHAR NOT NULL,
        value,
        product VARCHAR NOT NULL,
        hdu_index INTEGER NOT NULL,
        FOREIGN KEY(product) REFERENCES products(product),
        FOREIGN KEY(product, hdu_index) REFERENCES hdus(product, hdu_index)
        )"""
# type: str
