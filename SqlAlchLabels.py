import pdart.add_pds_tools
import picmaker

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pdart.pds4.Archives import get_any_archive
from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LID import LID
import pdart.pds4.Product as P
from pdart.xml.Pretty import pretty_print
from pdart.xml.Schema import verify_label_or_raise
from pdart.xml.Templates import interpret_document_template

from SqlAlchTables import FitsProduct
from SqlAlchXml import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    from pdart.xml.Templates import DocTemplate


_NEW_DATABASE_NAME = 'sqlalch-database.db'
# type: str
# TODO This is cut-and-pasted from SqlAlch.  Refactor and remove.


PRODUCT_LID = 'urn:nasa:pds:hst_14334:data_wfc3_raw:icwy08q3q_raw'
# type: str


def bundle_database_filepath(bundle):
    # type: (B.Bundle) -> unicode

    # TODO This is cut-and-pasted from SqlAlch.  Refactor and remove.
    return os.path.join(bundle.absolute_filepath(), _NEW_DATABASE_NAME)


def ensure_directory(dir):
    # type: (AnyStr) -> None
    """Make the directory if it doesn't already exist."""

    # TODO This is cut-and-pasted from
    # pdart.pds4label.BrowseProductImageReduction.  Refactor and
    # remove.
    try:
        os.mkdir(dir)
    except OSError:
        pass
    assert os.path.isdir(dir), dir


##############################

_product_browse_template = interpret_document_template(
    """<?xml version="1.0"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Browse xmlns="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
                       xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                           http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">
<NODE name="Identification_Area" />
<NODE name="File_Area_Browse" />
</Product_Browse>""")
# type: DocTemplate


def make_product_browse_label(collection, product):
    # type: (Collection, Product) -> str
    logical_identifier = make_logical_identifier(str(product.lid))
    version_id = make_version_id()

    bundle = collection.bundle
    proposal_id = cast(int, bundle.proposal_id)
    text = ('This product contains a browse image of a %s image obtained by ' +
            'HST Observing Program %d.') % (str(collection.suffix).upper(),
                                            proposal_id)
    title = make_title(text)

    information_model_version = make_information_model_version()
    product_class = make_product_class('Product_Browse')

    label = _product_browse_template({
            'Identification_Area': make_identification_area(
                logical_identifier,
                version_id,
                title,
                information_model_version,
                product_class,
                combine_nodes_into_fragment([])),
            'File_Area_Browse': make_file_area_browse(product)
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        print label
        raise
    return pretty


##############################

_product_observational_template = interpret_document_template(
    """<?xml version="1.0"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
                       xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                           http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">
<NODE name="Identification_Area" />
<NODE name="Observation_Area" />
<NODE name="File_Area_Observational" />
</Product_Observational>
""")
# type: DocTemplate


def make_product_observational_label(product):
    # type: (Product) -> str

    logical_identifier = make_logical_identifier(str(product.lid))
    version_id = make_version_id()

    collection = product.collection
    bundle = collection.bundle
    proposal_id = cast(int, bundle.proposal_id)
    text = ('This product contains the %s image obtained by ' +
            'HST Observing Program %d.') % (str(collection.suffix).upper(),
                                            proposal_id)
    title = make_title(text)

    information_model_version = make_information_model_version()
    product_type = str(product.type)
    if product_type == 'fits_product':
        text = 'Product_Observational'
    else:
        assert False, 'Unimplemented for ' + product_type
    product_class = make_product_class(text)

    label = _product_observational_template({
            'Identification_Area': make_identification_area(
                logical_identifier,
                version_id,
                title,
                information_model_version,
                product_class,
                combine_nodes_into_fragment([])),
            'Observation_Area': make_observation_area(product),
            'File_Area_Observational': make_file_area_observational(product)
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        print label
        raise
    return pretty


##############################

_product_collection_template = interpret_document_template(
    """<?xml version="1.0"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Collection xmlns="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
                       xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                           http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">
<NODE name="Identification_Area" />
<NODE name="Collection" />
<NODE name="File_Area_Inventory" />
</Product_Collection>
""")
# type: DocTemplate


def make_product_collection_label(collection):
    # type: (Collection) -> str

    logical_identifier = make_logical_identifier(str(collection.lid))
    version_id = make_version_id()

    proposal_id = cast(int, collection.bundle.proposal_id)
    text = ('This collection contains the %s images obtained by ' +
            'HST Observing Program %d.') % (str(collection.suffix).upper(),
                                            proposal_id)

    title = make_title(text)
    information_model_version = make_information_model_version()
    product_class = make_product_class('Product_Collection')

    publication_year = '2000'  # TODO
    description = 'TODO'  # TODO
    citation_information = combine_nodes_into_fragment([
            make_citation_information(publication_year,
                                      description)
            ])

    label = _product_collection_template({
            'Identification_Area': make_identification_area(
                logical_identifier,
                version_id,
                title,
                information_model_version,
                product_class,
                citation_information),
            'Collection': make_collection(collection),
            'File_Area_Inventory': make_file_area_inventory(collection)
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        print label
        raise
    return pretty


##############################

_product_bundle_template = interpret_document_template(
    """<?xml version="1.0"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Bundle xmlns="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
                       xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                           http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">
<NODE name="Identification_Area"/>
<NODE name="Bundle"/>
<FRAGMENT name="Bundle_Member_Entry"/>
</Product_Bundle>
""")
# type: DocTemplate


def make_product_bundle_label(bundle):
    # type: (Bundle) -> str
    logical_identifier = make_logical_identifier(str(bundle.lid))
    version_id = make_version_id()

    proposal_id = cast(int, bundle.proposal_id)
    text = "This bundle contains images obtained from " + \
        "HST Observing Program %d." % proposal_id
    title = make_title(text)
    information_model_version = make_information_model_version()
    product_class = make_product_class('Product_Bundle')

    publication_year = '2000'  # TODO
    description = 'TODO'  # TODO

    label = _product_bundle_template({
            'Identification_Area': make_identification_area(
                logical_identifier,
                version_id,
                title,
                information_model_version,
                product_class,
                combine_nodes_into_fragment([
                        make_citation_information(publication_year,
                                                  description)
                        ])
                ),
            'Bundle': make_bundle(bundle),
            'Bundle_Member_Entry': combine_nodes_into_fragment(
                [make_bundle_member_entry(coll)
                 for coll in bundle.collections])
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        print label
        raise
    return pretty


##############################

_product_document_template = interpret_document_template(
    """<?xml version="1.0"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Document xmlns="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
                       xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                           http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">
<NODE name="Identification_Area"/>
<NODE name="Document"/>
</Product_Document>
""")
# type: DocTemplate


def make_product_document_label(bundle, product):
    # type: (B.Bundle, P.Product) -> str

    # TODO We're based for now on pdart.pds4 Bundle and Product, not
    # the DB versions.  Fix this.

    logical_identifier = make_logical_identifier(str(product.lid))
    version_id = make_version_id()

    proposal_id = cast(int, bundle.proposal_id())
    text = "This bundle contains images obtained from " + \
        "HST Observing Program %d." % proposal_id
    title = make_title(text)
    information_model_version = make_information_model_version()
    product_class = make_product_class('Product_Document')

    publication_date = '2000-01-01'  # TODO
    publication_year = '2000'  # TODO
    description = 'TODO'  # TODO
    files = [(u'bob', u'PDF')]

    label = _product_document_template({
            'Identification_Area': make_identification_area(
                logical_identifier,
                version_id,
                title,
                information_model_version,
                product_class,
                combine_nodes_into_fragment([
                        make_citation_information(publication_year,
                                                  description)
                        ])
                ),
            'Document': make_document(publication_date, files)
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        print label
        raise
    return pretty


##############################

_product_spice_kernel_template = interpret_document_template(
    """<?xml version="1.0"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_SPICE_Kernel xmlns="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
                       xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                           http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">
<NODE name="Identification_Area"/>
<NODE name="Context_Area"/>
<NODE name="Reference_List"/>
<NODE name="File_Area_SPICE_Kernel"/>
</Product_SPICE_Kernel>
""")
# type: DocTemplate


def make_product_spice_kernel_label(bundle, product, fits_product):
    # type: (B.Bundle, P.Product, Product) -> str

    # TODO We're based for now on pdart.pds4 Bundle and Product, not
    # the DB versions.  Fix this.

    logical_identifier = make_logical_identifier(str(product.lid))
    version_id = make_version_id()

    proposal_id = cast(int, bundle.proposal_id())
    text = "This product contains SPICE kernels for " + \
        "HST Observing Program %d." % proposal_id  # TODO
    title = make_title(text)
    information_model_version = make_information_model_version()
    product_class = make_product_class('Product_SPICE_Kernel')

    publication_date = '2000-01-01'  # TODO
    publication_year = '2000'  # TODO
    description = 'TODO'  # TODO
    files = [('bob', 'PDF')]

    label = _product_spice_kernel_template({
            'Identification_Area': make_identification_area(
                logical_identifier,
                version_id,
                title,
                information_model_version,
                product_class,
                combine_nodes_into_fragment([
                        make_citation_information(publication_year,
                                                  description)
                        ])
                ),
            'Context_Area': make_context_area(fits_product),
            'Reference_List': make_reference_list(),
            'File_Area_SPICE_Kernel': make_file_area_spice_kernel('bob')
            # TODO
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        print label
        raise
    return pretty


##############################


def make_browse_product(fits_product, browse_product):
    # type: (P.Product, P.Product) -> None
    file = list(fits_product.files())[0]
    basename = os.path.basename(file.full_filepath())
    basename = os.path.splitext(basename)[0] + '.jpg'
    browse_collection_dir = browse_product.collection().absolute_filepath()
    ensure_directory(browse_collection_dir)

    visit = HstFilename(basename).visit()
    target_dir = os.path.join(browse_collection_dir, ('visit_%s' % visit))
    ensure_directory(target_dir)

    picmaker.ImagesToPics([file.full_filepath()],
                          target_dir,
                          filter="None",
                          percentiles=(1, 99))


def make_db_browse_collection(session, browse_collection):
    # type: (Session, C.Collection) -> Collection

    # TODO Does Collection need to change to a more specific
    # BrowseCollection type?
    lid = str(browse_collection.lid)

    # TODO I'm deleting any previous record here, but only during
    # development.
    session.query(Collection).filter_by(lid=lid).delete()

    bundle = browse_collection.bundle()

    db_browse_collection = Collection(
        lid=lid,
        bundle_lid=str(bundle.lid),
        prefix=browse_collection.prefix(),
        suffix=browse_collection.suffix(),
        instrument=browse_collection.instrument(),
        full_filepath=browse_collection.absolute_filepath(),
        label_filepath=browse_collection.label_filepath(),
        inventory_name=browse_collection.inventory_name(),
        inventory_filepath=browse_collection.inventory_filepath())
    session.add(db_browse_collection)
    session.commit()
    return db_browse_collection


def make_db_browse_product(session, fits_product, browse_product):
    # type: (Session, P.Product, P.Product) -> Tuple[Collection, BrowseProduct]

    lid = str(browse_product.lid)

    # TODO I'm deleting any previous record here, but only during
    # development.
    session.query(BrowseProduct).filter_by(product_lid=lid).delete()
    session.query(Product).filter_by(lid=lid).delete()

    db_browse_product = BrowseProduct(
        lid=str(browse_product.lid),
        collection_lid=str(browse_product.collection().lid),
        label_filepath=browse_product.label_filepath(),
        browse_filepath=browse_product.absolute_filepath()
        )
    session.add(db_browse_product)
    session.commit()

    db_browse_collection = \
        make_db_browse_collection(session, browse_product.collection())

    return (db_browse_collection, db_browse_product)


def run():
    archive = get_any_archive()
    fits_product = P.Product(archive, LID(PRODUCT_LID))
    collection = fits_product.collection()
    bundle = fits_product.bundle()
    db_fp = bundle_database_filepath(bundle)
    print db_fp
    engine = create_engine('sqlite:///' + db_fp)

    session = sessionmaker(bind=engine)()
    # type: Session

    if False:
        db_fits_product = \
            session.query(Product).filter_by(lid=PRODUCT_LID).first()

        label = make_product_observational_label(db_fits_product)
        print label
        verify_label_or_raise(label)

    if False:
        browse_product = fits_product.browse_product()
        # three goals:
        # make browse_product in file system
        make_browse_product(fits_product, browse_product)
        # make browse_product in DB
        (db_browse_collection,
         db_browse_product) = make_db_browse_product(session,
                                                     fits_product,
                                                     browse_product)
        # make label
        label = make_product_browse_label(db_browse_collection,
                                          db_browse_product)
        print label
        verify_label_or_raise(label)

    # TODO Build inventory

    if False:
        COLLECTION_LID = str(collection.lid)
        db_collection = \
            session.query(Collection).filter_by(lid=COLLECTION_LID).first()

        label = make_product_collection_label(db_collection)
        print label
        verify_label_or_raise(label)

    if False:
        BUNDLE_LID = str(bundle.lid)
        db_bundle = \
            session.query(Bundle).filter_by(lid=BUNDLE_LID).first()

        label = make_product_bundle_label(db_bundle)
        print label
        verify_label_or_raise(label)

    if False:
        DOCUMENT_PRODUCT_LID = str(bundle.lid) + ":document:phase2"
        print DOCUMENT_PRODUCT_LID
        label = make_product_document_label(
            bundle, P.Product(archive, LID(DOCUMENT_PRODUCT_LID)))
        print label
        verify_label_or_raise(label)

    if True:
        SPICE_KERNEL_PRODUCT_LID = str(bundle.lid) + ":spice:kernel"
        # TODO The LID is legal but wrong, a placeholder
        print SPICE_KERNEL_PRODUCT_LID
        fits_product = \
            session.query(FitsProduct).filter_by(lid=PRODUCT_LID).first()
        assert fits_product, 'fits_product'
        label = make_product_spice_kernel_label(
            bundle,
            P.Product(archive, LID(SPICE_KERNEL_PRODUCT_LID)),
            fits_product)
        print label
        verify_label_or_raise(label)


if __name__ == '__main__':
    run()
