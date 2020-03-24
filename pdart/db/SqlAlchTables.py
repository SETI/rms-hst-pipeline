from typing import Any

from sqlalchemy import Column, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import RelationshipProperty, backref, relationship

Base: Any = declarative_base()


def create_tables(engine: Engine) -> None:
    Base.metadata.create_all(engine)


class Bundle(Base):
    __tablename__ = "bundles"
    lidvid = Column(String, primary_key=True, nullable=False)
    proposal_id = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"Bundle(lidvid={self.lidvid!r}, proposal_id={self.proposal_id})"


############################################################


class Collection(Base):
    __tablename__ = "collections"

    lidvid = Column(String, primary_key=True, nullable=False)
    bundle_lidvid = Column(
        String, ForeignKey("bundles.lidvid"), nullable=False, index=True
    )
    type = Column(String(24), nullable=False)

    __mapper_args__ = {"polymorphic_identity": "collection", "polymorphic_on": type}


class DocumentCollection(Collection):
    __tablename__ = "document_collections"

    collection_lidvid = Column(
        String, ForeignKey("collections.lidvid"), primary_key=True, nullable=False
    )

    __mapper_args__ = {
        "polymorphic_identity": "document_collection",
    }

    def __repr__(self) -> str:
        return f"DocumentCollection(lidvid={self.lidvid!r}, bundle_lidvid={self.bundle_lidvid!r})"


class NonDocumentCollection(Collection):
    __tablename__ = "non_document_collections"

    collection_lidvid = Column(
        String, ForeignKey("collections.lidvid"), primary_key=True, nullable=False
    )
    # eight is overkill, but that's fine
    instrument = Column(String(8), nullable=False)
    prefix = Column(String(8), nullable=False)
    suffix = Column(String(8), nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "non_document_collection",
    }

    def __repr__(self) -> str:
        return (
            f"NonDocumentCollection(lidvid={self.lidvid!r}, "
            f"bundle_lidvid={self.bundle_lidvid!r}, "
            f"instrument={self.instrument!r}, "
            f"prefix={self.prefix!r}, suffix={self.suffix!r})"
        )


############################################################


class Product(Base):
    __tablename__ = "products"

    lidvid = Column(String, primary_key=True, nullable=False)
    collection_lidvid = Column(
        String, ForeignKey("collections.lidvid"), nullable=False, index=True
    )
    type = Column(String(16), nullable=False)

    __mapper_args__ = {"polymorphic_identity": "product", "polymorphic_on": type}


class BrowseProduct(Product):
    """
    A database representation of a PDS4 product consisting of browse
    images.
    """

    __tablename__ = "browse_products"

    product_lidvid = Column(
        String, ForeignKey("products.lidvid"), primary_key=True, nullable=False
    )
    fits_product_lidvid = Column(
        String, ForeignKey("fits_products.product_lidvid"), nullable=False
    )

    __mapper_args__ = {
        "polymorphic_identity": "browse_product",
    }

    def __repr__(self) -> str:
        return (
            f"BrowseProduct(lidvid={self.lidvid!r}, "
            f"collection_lidvid={self.collection_lidvid!r}, "
            f"fits_product_lidvid={self.fits_product_lidvid!r})"
        )


class DocumentProduct(Product):
    """
    A database representation of a PDS4 product consisting of
    documents.
    """

    __tablename__ = "document_products"

    product_lidvid = Column(
        String, ForeignKey("products.lidvid"), primary_key=True, nullable=False
    )

    __mapper_args__ = {
        "polymorphic_identity": "document_product",
    }

    def __repr__(self) -> str:
        return (
            f"DocumentProduct(lidvid={self.lidvid!r}, "
            f"collection_lidvid={self.collection_lidvid!r})"
        )


class FitsProduct(Product):
    """
    A database representation of a PDS4 observational product
    consisting of a single FITS file.
    """

    __tablename__ = "fits_products"

    product_lidvid = Column(
        String, ForeignKey("products.lidvid"), primary_key=True, nullable=False
    )

    __mapper_args__ = {
        "polymorphic_identity": "fits_product",
    }

    def __repr__(self) -> str:
        return (
            f"FitsProduct(lidvid={self.lidvid!r}, "
            f"collection_lidvid={self.collection_lidvid!r})"
        )


############################################################


class File(Base):
    """
    A database representation of a single file that belongs to a
    product.
    """

    __tablename__ = "files"

    id = Column(Integer, primary_key=True, nullable=False)
    product_lidvid = Column(String, ForeignKey("products.lidvid"), nullable=False)
    basename = Column(String, nullable=False)
    type = Column(String(16), nullable=False)
    md5_hash = Column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("product_lidvid", "basename"),
        Index("idx_product_lidvid_basename", "product_lidvid", "basename"),
    )
    __mapper_args__ = {"polymorphic_identity": "file", "polymorphic_on": type}


class BadFitsFile(File):
    """
    A database representation of a FITS file belonging to a product
    that could not be read.
    """

    __tablename__ = "bad_fits_files"

    file_id = Column(Integer, ForeignKey("files.id"), primary_key=True, nullable=False)
    exception_message = Column(String, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "bad_fits_file"}

    def __repr__(self) -> str:
        return (
            f"BadFitsFile(id={self.id!r}, "
            f"product_lidvid={self.product_lidvid!r}, "
            f"basename={self.basename!r})"
        )


class BrowseFile(File):
    """
    A database representation of a browse file belonging to a browse product.
    """

    __tablename__ = "browse_files"

    file_id = Column(Integer, ForeignKey("files.id"), primary_key=True, nullable=False)
    byte_size = Column(Integer, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "browse_file"}

    def __repr__(self) -> str:
        return (
            f"BrowseFile(id={self.id}, "
            f"product_lidvid={self.product_lidvid!r}, "
            f"basename={self.basename!r}, "
            f"byte_size={self.byte_size})"
        )


class DocumentFile(File):
    """
    A database representation of a document file belonging to a
    product.
    """

    __tablename__ = "document_files"

    file_id = Column(Integer, ForeignKey("files.id"), primary_key=True, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "document_file"}

    def __repr__(self) -> str:
        return (
            f"DocumentFile(id={self.id}, "
            f"product_lidvid={self.product_lidvid!r}, "
            f"basename={self.basename!r})"
        )


class FitsFile(File):
    """
    A database representation of a FITS file belonging to a product.
    """

    __tablename__ = "fits_files"

    file_id = Column(Integer, ForeignKey("files.id"), primary_key=True, nullable=False)
    hdu_count = Column(Integer, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "fits_file"}

    def __repr__(self) -> str:
        return (
            f"FitsFile(id={self.id}, "
            f"product_lidvid={self.product_lidvid!r}, "
            f"basename={self.basename!r}, "
            f"hdu_count={self.hdu_count})"
        )


############################################################


class Hdu(Base):
    """
    A database representation of a FITS HDU.
    """

    __tablename__ = "hdus"

    product_lidvid = Column(
        String,
        ForeignKey("products.lidvid"),
        primary_key=True,
        nullable=False,
        index=True,
    )
    hdu_index = Column(Integer, primary_key=True, nullable=False)
    """The zero-based index of this HDU within its product's FITS file."""
    hdr_loc = Column(Integer, nullable=False)
    """The starting byte location of the header in the file"""
    dat_loc = Column(Integer, nullable=False)
    """The starting byte location of the data block in the file"""
    dat_span = Column(Integer, nullable=False)
    """The data size including padding"""

    product: RelationshipProperty = relationship(
        "FitsProduct", backref=backref("hdus", order_by=hdu_index)
    )

    def __repr__(self) -> str:
        return (
            f"Hdu(product_lid={self.product.lid!r}, "
            f"hdu_index={self.hdu_index}, "
            f"hdr_loc={self.hdr_loc}, "
            f"dat_loc={self.dat_loc}, "
            f"dat_span={self.dat_span})"
        )


class Card(Base):
    """
    A database representation of a card within an HDU of a FITS file.
    """

    __tablename__ = "cards"

    id = Column(Integer, primary_key=True, nullable=False)
    product_lidvid = Column(String, ForeignKey("products.lidvid"), nullable=False)
    card_index = Column(Integer, nullable=False)
    hdu_index = Column(Integer, ForeignKey("hdus.hdu_index"), nullable=False)
    keyword = Column(String, nullable=False)
    value = Column(String, nullable=True)

    hdu: RelationshipProperty = relationship(
        "Hdu", backref=backref("cards", order_by=id)
    )

    def __repr__(self) -> str:
        return (
            f"Card(product_lidvid={self.product_lidvid!r}, "
            f"hdu_index={self.hdu_index}, "
            f"card_index={self.card_index}, "
            f"keyword={self.keyword!r}, "
            f"value={self.value!r})"
        )


Index("idx_cards_product_hdu_index", Card.product_lidvid, Card.hdu_index)


############################################################


class BundleLabel(Base):
    """
    A database representation of a PDS4 bundle label.
    """

    __tablename__ = "bundle_labels"

    bundle_lidvid = Column(
        String, ForeignKey("bundles.lidvid"), primary_key=True, nullable=False
    )
    basename = Column(String, nullable=False)
    md5_hash = Column(String(32), nullable=False)


class CollectionLabel(Base):
    """
    A database representation of a PDS4 collection label.
    """

    __tablename__ = "collection_labels"

    collection_lidvid = Column(
        String, ForeignKey("collections.lidvid"), primary_key=True, nullable=False
    )
    basename = Column(String, nullable=False)
    md5_hash = Column(String(32), nullable=False)


class CollectionInventory(Base):
    """
    A database representation of a PDS4 collection inventory.
    """

    __tablename__ = "collection_inventories"

    collection_lidvid = Column(
        String, ForeignKey("collections.lidvid"), primary_key=True, nullable=False
    )
    basename = Column(String, nullable=False)
    md5_hash = Column(String(32), nullable=False)


class ProductLabel(Base):
    """
    A database representation of a PDS4 product label.
    """

    __tablename__ = "product_labels"

    product_lidvid = Column(
        String, ForeignKey("products.lidvid"), primary_key=True, nullable=False
    )
    basename = Column(String, nullable=False)
    md5_hash = Column(String(32), nullable=False)


############################################################


class ProposalInfo(Base):
    """
    Proposal-related information that should not change over time,
    organized by LID rather than LIDVID.
    """

    __tablename__ = "proposal_info"

    bundle_lid = Column(String, primary_key=True, nullable=False)
    proposal_title = Column(String, nullable=False)
    pi_name = Column(String, nullable=False)
    author_list = Column(String, nullable=False)
    proposal_year = Column(String, nullable=False)
    publication_year = Column(String, nullable=False)
