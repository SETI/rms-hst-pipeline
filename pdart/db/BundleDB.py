import os.path
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker

from pdart.db.SqlAlchTables import (
    Association,
    BadFitsFile,
    BrowseFile,
    BrowseProduct,
    Bundle,
    BundleLabel,
    Card,
    Collection,
    CollectionInventory,
    CollectionLabel,
    ContextCollection,
    ContextProduct,
    DocumentCollection,
    DocumentFile,
    DocumentProduct,
    File,
    FitsFile,
    FitsProduct,
    Hdu,
    OtherCollection,
    Product,
    ProductLabel,
    ProposalInfo,
    SchemaCollection,
    SchemaProduct,
    create_tables,
    switch_on_collection_subtype,
)
from pdart.db.Utils import file_md5
from pdart.labels.RawSuffixes import RAW_SUFFIXES
from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID

_BUNDLE_DB_NAME: str = "bundle$database.db"
_BUNDLE_DIRECTORY_PATTERN: str = r"\Ahst_([0-9]{5})\Z"
_COLLECTION_DIRECTORY_PATTERN: str = r"\A(([a-z]+)_([a-z0-9]+)_([a-z0-9_]+)|document)\Z"


def _get_other_suffixed_basename(filepath: str, suffix: str) -> str:
    # TODO BUFFALO A hack.  Make this private and refactor as necessary.
    match = re.match(r"\A([^_]+)_[^\.]+\..*\Z", os.path.basename(filepath))
    assert match
    return f"{match.group(1)}_{suffix}.fits"


def _get_shm_basename(filepath: str) -> str:
    # TODO BUFFALO A hack.  Make this private and refactor as necessary.
    # match = re.match(r"\A([^_]+)_[^\.]+\..*\Z", os.path.basename(filepath))
    # assert match
    # return str(match.group(1)) + "_shm.fits"

    return _get_other_suffixed_basename(filepath, "shm")


def _get_other_suffixed_product_lidvid(lidvid_str: str, suffix: str) -> str:
    lidvid = LIDVID(lidvid_str)
    lid = lidvid.lid()
    vid = lidvid.vid()
    other_lid = lid.to_other_suffixed_lid(suffix)
    # TODO BUFFALO A hack: this is only valid for the initial version.
    return str(LIDVID.create_from_lid_and_vid(other_lid, vid))


def _get_shm_product_lidvid(lidvid_str: str) -> str:
    # lidvid = LIDVID(lidvid_str)
    # lid = lidvid.lid()
    # vid = lidvid.vid()
    # shm_lid = lid.to_shm_lid()
    # # TODO BUFFALO A hack: this is only valid for the initial version.
    # return str(LIDVID.create_from_lid_and_vid(shm_lid, vid))
    return _get_other_suffixed_product_lidvid(lidvid_str, "shm")


def create_bundle_db_from_os_filepath(os_filepath: str) -> "BundleDB":
    return BundleDB("sqlite:///" + os_filepath)


def create_bundle_db_in_memory() -> "BundleDB":
    return BundleDB("sqlite:///")


def _sure_match(pattern: str, string: Optional[str], group_num: int) -> str:
    assert string is not None
    match = re.match(pattern, string)
    assert match is not None
    res = match.group(group_num)
    assert res is not None
    return res


def _lidvid_to_proposal_id(bundle_lidvid: str) -> int:
    lid = LIDVID(bundle_lidvid).lid()
    bundle_id = lid.bundle_id
    return int(_sure_match(_BUNDLE_DIRECTORY_PATTERN, bundle_id, 1))


def _lidvid_to_instrument(other_collection_lidvid: str) -> str:
    lid = LIDVID(other_collection_lidvid).lid()
    collection_id = lid.collection_id
    return _sure_match(_COLLECTION_DIRECTORY_PATTERN, collection_id, 3)


def _lidvid_to_prefix(other_collection_lidvid: str) -> str:
    lid = LIDVID(other_collection_lidvid).lid()
    collection_id = lid.collection_id
    return _sure_match(_COLLECTION_DIRECTORY_PATTERN, collection_id, 2)


def _lidvid_to_suffix(other_collection_lidvid: str) -> str:
    lid = LIDVID(other_collection_lidvid).lid()
    collection_id = lid.collection_id
    return _sure_match(_COLLECTION_DIRECTORY_PATTERN, collection_id, 4)


class BundleDB(object):
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine = create_engine(url)
        self.session = sessionmaker(bind=self.engine)()

    def dump(self) -> None:
        for line in self.engine.raw_connection().iterdump():
            print(line)

    def create_tables(self) -> None:
        create_tables(self.engine)

    ############################################################

    def create_bundle(self, bundle_lidvid: str) -> None:
        """
        Create a bundle with this LIDVID if none exists.
        """

        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if not self.bundle_exists(bundle_lidvid):
            proposal_id = _lidvid_to_proposal_id(bundle_lidvid)
            self.session.add(Bundle(lidvid=bundle_lidvid, proposal_id=proposal_id))
            self.session.commit()

    def bundle_exists(self, bundle_lidvid: str) -> bool:
        """
        Returns True iff a bundle with the given LIDVID exists in the database.
        """
        return self.session.query(
            exists().where(Bundle.lidvid == bundle_lidvid)
        ).scalar()

    def get_bundle(self) -> Bundle:
        return self.session.query(Bundle).one()

    def get_bundle_collections(self, bundle_lidvid: str) -> List[Collection]:
        return (
            self.session.query(Collection)
            .filter(Collection.bundle_lidvid == bundle_lidvid)
            .order_by(Collection.lidvid)
            .all()
        )

    ############################################################

    def create_context_collection(
        self, collection_lidvid: str, bundle_lidvid: str
    ) -> None:
        """
        Create a context collection with this LIDVID if none exists.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if self.collection_exists(collection_lidvid):
            collection = self.get_collection(collection_lidvid)
            context_collection_exists = switch_on_collection_subtype(
                collection, True, False, False, False
            )
            if context_collection_exists:
                pass
            else:
                raise Exception(
                    f"non-context collection with "
                    f"LIDVID {collection_lidvid} already exists"
                )
        else:
            self.session.add(
                ContextCollection(lidvid=collection_lidvid, bundle_lidvid=bundle_lidvid)
            )
            self.session.commit()

    def create_document_collection(
        self, collection_lidvid: str, bundle_lidvid: str
    ) -> None:
        """
        Create a document_collection with this LIDVID if none exists.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if self.collection_exists(collection_lidvid):
            collection = self.get_collection(collection_lidvid)
            document_collection_exists = switch_on_collection_subtype(
                collection, False, True, False, False
            )
            if document_collection_exists:
                pass
            else:
                raise Exception(
                    f"non-document collection with "
                    f"LIDVID {collection_lidvid} already exists"
                )
        else:
            self.session.add(
                DocumentCollection(
                    lidvid=collection_lidvid, bundle_lidvid=bundle_lidvid
                )
            )
            self.session.commit()

    def create_schema_collection(
        self, collection_lidvid: str, bundle_lidvid: str
    ) -> None:
        """
        Create a schema_collection with this LIDVID if none exists.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid(), collection_lidvid
        assert LIDVID(bundle_lidvid).is_bundle_lidvid(), bundle_lidvid
        if self.collection_exists(collection_lidvid):
            collection = self.get_collection(collection_lidvid)
            schema_collection_exists = switch_on_collection_subtype(
                collection, False, False, True, False
            )
            if schema_collection_exists:
                pass
            else:
                raise Exception(
                    f"non-schema collection with "
                    f"LIDVID {collection_lidvid} already exists"
                )
        else:
            self.session.add(
                SchemaCollection(lidvid=collection_lidvid, bundle_lidvid=bundle_lidvid)
            )
            self.session.commit()

    def create_other_collection(
        self, collection_lidvid: str, bundle_lidvid: str
    ) -> None:
        """
        Create an other_collection with this LIDVID if none exists.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if self.collection_exists(collection_lidvid):
            collection = self.get_collection(collection_lidvid)
            other_collection_exists = switch_on_collection_subtype(
                collection, False, False, False, True
            )
            if other_collection_exists:
                # it already exists
                pass
            else:
                raise Exception(
                    f"non-other-collection with "
                    f"LIDVID {collection_lidvid} already exists"
                )
        else:
            instrument = _lidvid_to_instrument(collection_lidvid)
            prefix = _lidvid_to_prefix(collection_lidvid)
            suffix = _lidvid_to_suffix(collection_lidvid)
            self.session.add(
                OtherCollection(
                    lidvid=collection_lidvid,
                    collection_lidvid=collection_lidvid,
                    bundle_lidvid=bundle_lidvid,
                    instrument=instrument,
                    prefix=prefix,
                    suffix=suffix,
                )
            )
            self.session.commit()

    def collection_exists(self, collection_lidvid: str) -> bool:
        """
        Returns True iff a collection with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(Collection.lidvid == collection_lidvid)
        ).scalar()

    def get_collection(self, lidvid: str) -> Collection:
        return self.session.query(Collection).filter(Collection.lidvid == lidvid).one()

    def get_collection_products(self, collection_lidvid: str) -> List[Product]:
        return (
            self.session.query(Product)
            .filter(Product.collection_lidvid == collection_lidvid)
            .order_by(Product.lidvid)
            .all()
        )

    ############################################################

    def create_browse_product(
        self,
        browse_product_lidvid: str,
        fits_product_lidvid: str,
        collection_lidvid: str,
    ) -> None:
        """
        Create a product with this LIDVID if none exists.
        """
        assert LIDVID(browse_product_lidvid).is_product_lidvid()
        assert LIDVID(fits_product_lidvid).is_product_lidvid()
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.product_exists(fits_product_lidvid):
            if not self.fits_product_exists(fits_product_lidvid):
                raise Exception(f"product {fits_product_lidvid} is not a FITS product")
        else:
            raise Exception(
                f"FITS product {fits_product_lidvid} must exist before "
                f"building browse product {browse_product_lidvid}"
            )

        if self.product_exists(browse_product_lidvid):
            if self.browse_product_exists(browse_product_lidvid):
                pass
            else:
                raise Exception(
                    f"non-browse product with LIDVID {browse_product_lidvid} already exists"
                )
        else:
            self.session.add(
                BrowseProduct(
                    lidvid=browse_product_lidvid,
                    collection_lidvid=collection_lidvid,
                    fits_product_lidvid=fits_product_lidvid,
                )
            )
            self.session.commit()

    def create_document_product(
        self, product_lidvid: str, collection_lidvid: str
    ) -> None:
        """
        Create a product with this LIDVID if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.product_exists(product_lidvid):
            if self.document_product_exists(product_lidvid):
                pass
            else:
                raise Exception(
                    f"non-document product with LIDVID {product_lidvid} already exists"
                )
        else:
            self.session.add(
                DocumentProduct(
                    lidvid=product_lidvid, collection_lidvid=collection_lidvid
                )
            )
            self.session.commit()

    def create_fits_product(self, product_lidvid: str, collection_lidvid: str) -> None:
        """
        Create a product with this LIDVID if none exists.
        """
        product_lidvid2 = LIDVID(product_lidvid)
        assert product_lidvid2.is_product_lidvid(), product_lidvid
        rootname: str = cast(str, product_lidvid2.lid().product_id)
        LIDVID(collection_lidvid)
        if self.product_exists(product_lidvid):
            if self.fits_product_exists(product_lidvid):
                pass
            else:
                raise Exception(
                    f"non-FITS product with LIDVID {product_lidvid} already exists"
                )
        else:
            self.session.add(
                FitsProduct(
                    lidvid=product_lidvid,
                    collection_lidvid=collection_lidvid,
                    rootname=rootname,
                )
            )
            self.session.commit()

    def product_exists(self, product_lidvid: str) -> bool:
        """
        Returns True iff a product with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(Product.lidvid == product_lidvid)
        ).scalar()

    def browse_product_exists(self, product_lidvid: str) -> bool:
        """
        Returns True iff a browse product with the given LIDVID exists
        in the database.
        """
        return self.session.query(
            exists().where(BrowseProduct.product_lidvid == product_lidvid)
        ).scalar()

    def document_product_exists(self, product_lidvid: str) -> bool:
        """
        Returns True iff a document product with the given LIDVID
        exists in the database.
        """
        return self.session.query(
            exists().where(DocumentProduct.product_lidvid == product_lidvid)
        ).scalar()

    def fits_product_exists(self, product_lidvid: str) -> bool:
        """
        Returns True iff a FITS product with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(FitsProduct.product_lidvid == product_lidvid)
        ).scalar()

    def get_product(self, lidvid: str) -> Product:
        return self.session.query(Product).filter(Product.lidvid == lidvid).one()

    def get_product_file(self, product_lidvid: str) -> File:
        """When you know there's only one, as in browse and FITS products"""
        return (
            self.session.query(File).filter(File.product_lidvid == product_lidvid).one()
        )

    def get_product_files(self, product_lidvid: str) -> List[File]:
        return (
            self.session.query(File)
            .filter(File.product_lidvid == product_lidvid)
            .order_by(File.basename)
            .all()
        )

    ############################################################
    def create_context_product(self, id: str) -> None:
        """
        Create a context product with this LIDVID if none exists.
        """
        if "::" in id:
            assert LIDVID(id).is_product_lidvid()
        else:
            assert LID(id).is_product_lid()

        if not self.context_product_exists(id):
            self.session.add(ContextProduct(lidvid=id))
            self.session.commit()

    def context_product_exists(self, lidvid: str) -> bool:
        """
        Returns True iff a context product with the given LIDVID exists
        in the database.
        """
        return self.session.query(
            exists().where(ContextProduct.lidvid == lidvid)
        ).scalar()

    def get_context_products(self) -> List[ContextProduct]:
        return self.session.query(ContextProduct).order_by(ContextProduct.lidvid).all()

    ############################################################
    def create_schema_product(self, id: str) -> None:
        """
        Create a schema product with this LIDVID if none exists.
        """
        if "::" in id:
            assert LIDVID(id).is_product_lidvid()
        else:
            assert LID(id).is_product_lid()

        if not self.schema_product_exists(id):
            self.session.add(SchemaProduct(lidvid=id))
            self.session.commit()

    def schema_product_exists(self, lidvid: str) -> bool:
        """
        Returns True iff a schema product with the given LIDVID exists
        in the database.
        """
        return self.session.query(
            exists().where(SchemaProduct.lidvid == lidvid)
        ).scalar()

    def get_schema_products(self) -> List[SchemaProduct]:
        return self.session.query(SchemaProduct).order_by(SchemaProduct.lidvid).all()

    ############################################################

    def create_bad_fits_file(
        self,
        os_filepath: str,
        basename: str,
        product_lidvid: str,
        exception_message: str,
    ) -> None:
        """
        Create a bad FITS file record with this basename belonging to
        the product if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.fits_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                BadFitsFile(
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                    product_lidvid=product_lidvid,
                    exception_message=exception_message,
                )
            )
            self.session.commit()

    def create_browse_file(
        self, os_filepath: str, basename: str, product_lidvid: str, byte_size: int
    ) -> None:
        """
        Create a browse file with this basename belonging to the product
        if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.browse_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                BrowseFile(
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                    product_lidvid=product_lidvid,
                    byte_size=byte_size,
                )
            )
            self.session.commit()
            assert self.browse_file_exists(basename, product_lidvid)

    def create_document_file(
        self, os_filepath: str, basename: str, product_lidvid: str
    ) -> None:
        """
        Create a document file with this basename belonging to the product
        if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.document_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                DocumentFile(
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                    product_lidvid=product_lidvid,
                )
            )
            self.session.commit()
            assert self.document_file_exists(basename, product_lidvid)

    def create_fits_file(
        self, os_filepath: str, basename: str, product_lidvid: str, hdu_count: int
    ) -> None:
        """
        Create a FITS file with this basename belonging to the product
        if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.fits_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                FitsFile(
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                    product_lidvid=product_lidvid,
                    rootname=HstFilename(os_filepath).rootname(),
                    hdu_count=hdu_count,
                )
            )
            self.session.commit()
            assert self.fits_file_exists(basename, product_lidvid)

    def file_exists(self, basename: str, product_lidvid: str) -> bool:
        """
        Returns True iff a file with the given LIDVID and basename
        exists in the database.
        """
        return self.session.query(
            exists()
            .where(File.basename == basename)
            .where(File.product_lidvid == product_lidvid)
        ).scalar()

    def bad_fits_file_exists(self, basename: str, product_lidvid: str) -> bool:
        """
        Returns True iff a bad FITS file record with the given LIDVID
        and basename exists in the database.
        """
        return self.session.query(
            exists()
            .where(BadFitsFile.basename == basename)
            .where(BadFitsFile.product_lidvid == product_lidvid)
            .where(File.type == "bad_fits_file")
        ).scalar()

    def browse_file_exists(self, basename: str, product_lidvid: str) -> bool:
        """
        Returns True iff a browse file with the given LIDVID and
        basename exists in the database.
        """
        return self.session.query(
            exists()
            .where(BrowseFile.basename == basename)
            .where(BrowseFile.product_lidvid == product_lidvid)
            .where(File.type == "browse_file")
        ).scalar()

    def document_file_exists(self, basename: str, product_lidvid: str) -> bool:
        """
        Returns True iff a document file with the given LIDVID and
        basename exists in the database.
        """
        return self.session.query(
            exists()
            .where(DocumentFile.basename == basename)
            .where(DocumentFile.product_lidvid == product_lidvid)
            .where(File.type == "document_file")
        ).scalar()

    def fits_file_exists(self, basename: str, product_lidvid: str) -> bool:
        """
        Returns True iff a FITS file with the given LIDVID and
        basename exists in the database.
        """
        return self.session.query(
            exists()
            .where(FitsFile.basename == basename)
            .where(FitsFile.product_lidvid == product_lidvid)
            .where(File.type == "fits_file")
        ).scalar()

    def get_file(self, basename: str, product_lidvid: str) -> File:
        return (
            self.session.query(File)
            .filter(File.product_lidvid == product_lidvid, File.basename == basename)
            .one()
        )

    ############################################################

    # The pattern of creation and access function used in higher-level
    # objects (bundles, collections, products) you see above break
    # down at this point, since queries inside the FITS file are
    # handled differently.

    def hdu_exists(self, index: int, basename: str, product_lidvid: str) -> bool:
        """
        Returns True iff the n-th HDU for that FITS file exists
        """
        return self.session.query(
            exists()
            .where(Hdu.product_lidvid == product_lidvid)
            .where(Hdu.hdu_index == index)
        ).scalar()

    # TODO basename and index belong not to Hdu but its FitsFile.
    #    def get_hdu(self, index: int, basename: str, product_lidvid: str) -> Hdu:
    #        return (
    #            self.session.query(Hdu)
    #            .filter(
    #                Hdu.product_lidvid == product_lidvid,
    #                Hdu.basename == basename,
    #                Hdu.index == index,
    #            )
    #            .one()
    #        )

    def get_file_offsets(
        self, fits_product_lidvid: str
    ) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of 4-tuples of (hdu_index, hdr_loc, dat_loc and
        dat_span), one tuple for each HDU in the file.
        """

        hdus = (
            self.session.query(Hdu)
            .filter(Hdu.product_lidvid == fits_product_lidvid)
            .order_by(Hdu.hdu_index)
        )
        return [(hdu.hdu_index, hdu.hdr_loc, hdu.dat_loc, hdu.dat_span) for hdu in hdus]

    ############################################################

    def card_exists(self, keyword: str, hdu_index: int, product_lidvid: str) -> bool:
        """
        Returns True iff there is a card with the given keyword in
        the n-th HDU of the FITS file for that product.
        """
        return self.session.query(
            exists()
            .where(Card.product_lidvid == product_lidvid)
            .where(Card.hdu_index == hdu_index)
            .where(Card.keyword == keyword)
        ).scalar()

    def get_card_dictionaries(
        self, fits_product_lidvid: str, basename: str
    ) -> List[Dict[str, Any]]:
        """
        Return a list of dictionaries mapping FITS keys to their
        values, one per Hdu in the FITS file.
        """

        def get_card_dictionary(index: int) -> Dict[str, Any]:
            cards = (
                self.session.query(Card)
                .filter(Card.product_lidvid == fits_product_lidvid)
                .filter(Card.hdu_index == index)
            )
            return {card.keyword: card.value for card in cards}

        # TODO The cast is a hack.  How should it properly be done?
        file = cast(FitsFile, self.get_file(basename, fits_product_lidvid))
        return [get_card_dictionary(i) for i in range(file.hdu_count)]

    def get_other_suffixed_card_dictionaries(
        self, fits_product_lidvid: str, basename: str, suffix: str
    ) -> List[Dict[str, Any]]:
        """
        Return a list of dictionaries mapping FITS keys to their
        values, one per Hdu in the FITS file.
        """
        # TODO BUFFALO
        try:
            return self.get_card_dictionaries(
                _get_other_suffixed_product_lidvid(fits_product_lidvid, suffix),
                _get_other_suffixed_basename(basename, suffix),
            )
        except:
            return [{}]

    def get_other_suffixed_card_dictionaries_and_lidvid(
        self, fits_product_lidvid: str, basename: str, suffix: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Return a list of dictionaries mapping FITS keys to their
        values, one per Hdu in the FITS file.
        """
        # TODO BUFFALO
        other_lidvid = _get_other_suffixed_product_lidvid(fits_product_lidvid, suffix)

        try:
            return (
                other_lidvid,
                self.get_card_dictionaries(
                    other_lidvid, _get_other_suffixed_basename(basename, suffix)
                ),
            )

        except:
            return (other_lidvid, [{}])

    def get_raw_card_dictionaries(
        self, fits_product_lidvid: str, basename: str
    ) -> List[Dict[str, Any]]:
        """
        Return a list of dictionaries mapping FITS keys to their
        values, one per Hdu in the FITS file.
        """
        return self.get_other_suffixed_card_dictionaries(
            fits_product_lidvid, basename, "raw"
        )

    def get_shm_card_dictionaries(
        self, fits_product_lidvid: str, basename: str
    ) -> List[Dict[str, Any]]:
        """
        Return a list of dictionaries mapping FITS keys to their
        values, one per Hdu in the FITS file.
        """
        return self.get_other_suffixed_card_dictionaries(
            fits_product_lidvid, basename, "shm"
        )

    ############################################################

    def create_bundle_label(
        self, os_filepath: str, basename: str, bundle_lidvid: str
    ) -> None:
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if self.bundle_label_exists(bundle_lidvid):
            pass
        else:
            self.session.add(
                BundleLabel(
                    bundle_lidvid=bundle_lidvid,
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                )
            )
            self.session.commit()
            assert self.bundle_label_exists(bundle_lidvid)

    def bundle_label_exists(self, bundle_lidvid: str) -> bool:
        """
        Returns True iff there is a label for the bundle with the
        given LIDVID.
        """
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        return self.session.query(
            exists().where(BundleLabel.bundle_lidvid == bundle_lidvid)
        ).scalar()

    def get_bundle_label(self, bundle_lidvid: str) -> BundleLabel:
        """
        Returns the label for the bundle with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        return (
            self.session.query(BundleLabel)
            .filter(BundleLabel.bundle_lidvid == bundle_lidvid)
            .one()
        )

    ############################################################

    def create_collection_label(
        self, os_filepath: str, basename: str, collection_lidvid: str
    ) -> None:
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.collection_label_exists(collection_lidvid):
            pass
        else:
            self.session.add(
                CollectionLabel(
                    collection_lidvid=collection_lidvid,
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                )
            )
            self.session.commit()
            assert self.collection_label_exists(collection_lidvid)

    def collection_label_exists(self, collection_lidvid: str) -> bool:
        """
        Returns True iff there is a label for the collection with the
        given LIDVID.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return self.session.query(
            exists().where(CollectionLabel.collection_lidvid == collection_lidvid)
        ).scalar()

    def get_collection_label(self, collection_lidvid: str) -> CollectionLabel:
        """
        Returns the label for the collection with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return (
            self.session.query(CollectionLabel)
            .filter(CollectionLabel.collection_lidvid == collection_lidvid)
            .one()
        )

    ############################################################

    def create_collection_inventory(
        self, os_filepath: str, basename: str, collection_lidvid: str
    ) -> None:
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.collection_inventory_exists(collection_lidvid):
            pass
        else:
            self.session.add(
                CollectionInventory(
                    collection_lidvid=collection_lidvid,
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                )
            )
            self.session.commit()
            assert self.collection_inventory_exists(collection_lidvid)

    def collection_inventory_exists(self, collection_lidvid: str) -> bool:
        """
        Returns True iff there is a inventory for the collection with the
        given LIDVID.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return self.session.query(
            exists().where(CollectionInventory.collection_lidvid == collection_lidvid)
        ).scalar()

    def get_collection_inventory(self, collection_lidvid: str) -> CollectionInventory:
        """
        Returns the inventory for the collection with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return (
            self.session.query(CollectionInventory)
            .filter(CollectionInventory.collection_lidvid == collection_lidvid)
            .one()
        )

    ############################################################

    def create_product_label(
        self, os_filepath: str, basename: str, product_lidvid: str
    ) -> None:
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.product_label_exists(product_lidvid):
            pass
        else:
            self.session.add(
                ProductLabel(
                    product_lidvid=product_lidvid,
                    basename=basename,
                    md5_hash=file_md5(os_filepath),
                )
            )
            self.session.commit()
            assert self.product_label_exists(product_lidvid)

    def product_label_exists(self, product_lidvid: str) -> bool:
        """
        Returns True iff there is a label for the product with the
        given LIDVID.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        return self.session.query(
            exists().where(ProductLabel.product_lidvid == product_lidvid)
        ).scalar()

    def get_product_label(self, product_lidvid: str) -> ProductLabel:
        """
        Returns the label for the product with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        return (
            self.session.query(ProductLabel)
            .filter(ProductLabel.product_lidvid == product_lidvid)
            .one()
        )

    ############################################################

    def proposal_info_exists(self, bundle_lid: str) -> bool:
        assert LID(bundle_lid).is_bundle_lid()
        return self.session.query(
            exists().where(ProposalInfo.bundle_lid == bundle_lid)
        ).scalar()

    def create_proposal_info(
        self,
        bundle_lid: str,
        proposal_title: str,
        pi_name: str,
        author_list: str,
        proposal_year: str,
        publication_year: str,
    ) -> None:
        """
        Creates a record of proposal info for the given LID.

        NOTE: We don't allow updating through this interface now.  We
        might want to allow it in the future.
        """
        assert LID(bundle_lid).is_bundle_lid()
        if self.proposal_info_exists(bundle_lid):
            raise Exception(f"proposal info with LID {bundle_lid} already exists")
        else:
            self.session.add(
                ProposalInfo(
                    bundle_lid=bundle_lid,
                    proposal_title=proposal_title,
                    pi_name=pi_name,
                    author_list=author_list,
                    proposal_year=proposal_year,
                    publication_year=publication_year,
                )
            )

    def get_proposal_info(self, bundle_lid: str) -> ProposalInfo:
        return (
            self.session.query(ProposalInfo)
            .filter(ProposalInfo.bundle_lid == bundle_lid)
            .one()
        )

    ############################################################

    def close(self) -> None:
        """
        Close the session associated with this BundleDB.
        """
        self.session.close()
        self.session = None

    def is_open(self) -> bool:
        """
        Return True iff the session associated with this BundleDB has not
        been closed.
        """
        return self.session is not None

    ############################################################

    # the associations nightmare
    def get_associations(self, product_lidvid: str) -> List[Association]:
        return (
            self.session.query(Association)
            .filter(Association.product_lidvid == product_lidvid)
            .order_by(Association.association_index)
            .all()
        )

    def get_fits_products_by_rootname(self, rootname: str) -> List[FitsProduct]:
        return (
            self.session.query(FitsProduct)
            .filter(FitsProduct.rootname == rootname)
            .order_by(Product.lidvid)
            .all()
        )

    def get_associated_key_products(self, product_lidvid: str) -> List[FitsProduct]:
        return [
            fits_prod
            for association in self.get_associations(product_lidvid)
            for fits_prod in self.get_fits_products_by_rootname(
                association.memname.lower()
            )
            if self._is_key_product(fits_prod)
        ]

    def _is_key_product(self, fits_product: FitsProduct) -> bool:
        # A "key" product is a raw data file (with suffix RAW or C0F).
        collection_lidvid = fits_product.collection_lidvid
        collection = self.get_collection(collection_lidvid)
        is_other_collection = switch_on_collection_subtype(
            collection, False, False, False, True
        )
        return cast(OtherCollection, collection).suffix in RAW_SUFFIXES
