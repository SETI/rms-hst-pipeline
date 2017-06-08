"""
Creation of labels and writing them to the filesystem.  Creation of
browse products.
"""
import logging
import os.path

from pdart.db.SqlAlchTables import DocumentCollection, FitsProduct
from pdart.db.SqlAlchXml import *
import pdart.pds4.Product as P
from pdart.xml.Pretty import pretty_print
from pdart.xml.Templates import interpret_document_template

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pdart.pds4.Bundle as B
    from pdart.db.SqlAlchTables import DocumentProduct
    from pdart.xml.Templates import DocTemplate


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


def make_and_save_product_browse_label(collection, browse_product):
    # type: (Collection, BrowseProduct) -> str
    """
    Given the database Collection row and BrowseProduct row, create a
    product label and save it to the filesystem.
    """
    # PRECONDITION
    assert collection
    assert browse_product

    label = make_product_browse_label(collection, browse_product)
    label_filepath = str(browse_product.label_filepath)
    with open(label_filepath, "w") as f:
        f.write(label)

    # POSTCONDITION
    assert os.path.isfile(str(browse_product.label_filepath))

    return label


def make_product_browse_label(collection, browse_product):
    # type: (Collection, BrowseProduct) -> str
    """
    Given the database Collection row and BrowseProduct row, create a
    product label and return it.
    """
    # PRECONDITION
    assert collection
    assert browse_product

    logical_identifier = make_logical_identifier(str(browse_product.lid))
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
            'File_Area_Browse': make_file_area_browse(browse_product)
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        logging.getLogger(__name__).error(label)
        raise

    # POSTCONDITION
    assert pretty

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


def make_and_save_product_observational_label(fits_product):
    # type: (FitsProduct) -> str
    """
    Given the database Collection row and FitsProduct row, create a
    product label and save it to the filesystem.
    """
    # PRECONDITION
    assert fits_product

    label = make_product_observational_label(fits_product)
    label_filepath = str(fits_product.label_filepath)
    with open(label_filepath, "w") as f:
        f.write(label)

    # POSTCONDITION
    assert os.path.isfile(str(fits_product.label_filepath))

    return label


def make_product_observational_label(fits_product):
    # type: (FitsProduct) -> str
    """
    Given the database Collection row and FitsProduct row, create a
    product label and return it.
    """
    # PRECONDITION: FitsProduct exists
    assert fits_product

    logical_identifier = make_logical_identifier(str(fits_product.lid))
    version_id = make_version_id()

    collection = fits_product.collection
    bundle = collection.bundle
    proposal_id = cast(int, bundle.proposal_id)
    text = ('This product contains the %s image obtained by ' +
            'HST Observing Program %d.') % (str(collection.suffix).upper(),
                                            proposal_id)
    title = make_title(text)

    information_model_version = make_information_model_version()
    product_type = str(fits_product.type)
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
            'Observation_Area': make_observation_area(fits_product),
            'File_Area_Observational': make_file_area_observational(
                fits_product)
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        logging.getLogger(__name__).error(label)
        raise

    # POSTCONDITION
    assert pretty

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


def make_and_save_product_collection_label(collection):
    # type: (Collection) -> str
    """
    Given the database Collection row, create a collection label and
    save it to the filesystem.
    """
    # PRECONDITION
    assert collection
    # assert: that all the collection's products have matching
    # database entries including FITS, browse, SPICE kernel and
    # document products.  We can't really check this though.

    label = make_product_collection_label(collection)
    label_filepath = str(collection.label_filepath)
    with open(label_filepath, "w") as f:
        f.write(label)

    # POSTCONDITION
    assert os.path.isfile(str(collection.label_filepath))

    return label


def make_product_collection_label(collection):
    # type: (Collection) -> str
    """
    Given the database Collection row, create a collection label and
    return it.
    """
    # PRECONDITION
    assert collection
    # assert: that all the collection's products have matching
    # database entries including FITS, browse, SPICE kernel and
    # document products.  We can't really check this though.

    logical_identifier = make_logical_identifier(str(collection.lid))
    version_id = make_version_id()

    proposal_id = cast(int, collection.bundle.proposal_id)

    is_document_collection = isinstance(collection, DocumentCollection)
    if is_document_collection:
        text = ('This collection contains documentation for the '
                'HST Observing Program %d.') % (proposal_id)
    else:
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
        logging.getLogger(__name__).error(label)
        raise

    # POSTCONDITION
    assert pretty

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


def make_and_save_product_bundle_label(bundle):
    # type: (Bundle) -> str
    """
    Given the database Bundle row, create a bundle label and save it
    to the filesystem.
    """
    # PRECONDITION
    assert bundle

    label = make_product_bundle_label(bundle)
    label_filepath = str(bundle.label_filepath)
    with open(label_filepath, "w") as f:
        f.write(label)

    # POSTCONDITION
    assert os.path.isfile(str(bundle.label_filepath))

    return label


def make_product_bundle_label(bundle):
    # type: (Bundle) -> str
    """
    Given the database Bundle row, create a bundle label and return
    it.
    """
    # PRECONDITIONS
    assert bundle

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
        logging.getLogger(__name__).error(label)
        raise

    # POSTCONDITION
    assert pretty

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


def _make_file_name_std_pair(basename):
    # type: (AnyStr) -> Tuple[AnyStr, AnyStr]
    (root, ext) = os.path.splitext(basename)
    if ext == '.prop':
        return (basename, '7-Bit ASCII Text')
    elif ext == '.pro':
        return (basename, '7-Bit ASCII Text')
    elif ext == '.apt':
        return (basename, 'UTF-8 Text')
    elif ext == '.pdf':
        return (basename, 'PDF')
    else:
        raise Exception('Unknown document file extension %s in %s' %
                        (ext, basename))


def make_and_save_product_document_label(bundle, document_product):
    # type: (Bundle, DocumentProduct) -> str
    """
    Given the database Bundle row and DocumentProduct row, create a
    product label and save it to the filesystem.
    """
    # PRECONDITION
    assert document_product

    label = make_product_document_label(bundle, document_product)
    label_filepath = str(document_product.label_filepath)
    with open(label_filepath, "w") as f:
        f.write(label)

    # POSTCONDITION
    assert os.path.isfile(str(document_product.label_filepath))

    return label


def make_product_document_label(db_bundle, db_document_product):
    # type: (Bundle, DocumentProduct) -> str
    """
    Given the database Bundle row and DocumentProduct row, create a
    product label and return it.
    """
    # PRECONDITION
    assert db_document_product

    logical_identifier = make_logical_identifier(str(db_document_product.lid))
    version_id = make_version_id()

    proposal_id = cast(int, 0 + db_bundle.proposal_id)
    text = "This bundle contains images obtained from " + \
        "HST Observing Program %d." % proposal_id
    title = make_title(text)
    information_model_version = make_information_model_version()
    product_class = make_product_class('Product_Document')

    publication_date = '2000-01-01'  # TODO
    publication_year = '2000'  # TODO
    description = 'TODO'  # TODO
    short_files = ['' + file.file_basename
                   for file in db_document_product.document_files]
    # This should be tuples of file_name and std
    files = [_make_file_name_std_pair(f) for f in short_files]

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
        logging.getLogger(__name__).error(label)
        raise

    # POSTCONDITION
    assert pretty

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


def make_and_save_product_spice_kernel_label(bundle, product, fits_product):
    # type: (B.Bundle, P.Product, Product) -> str
    """
    Given the Bundle and Product objects and FitsProduct row, create a
    product label and save it to the filesystem.  TODO unimplemented
    """
    # PRECONDITION
    assert False, "ensure this implementation makes sense"  # it doesn't
    # assert spice_kernel exists

    label = _make_product_spice_kernel_label(bundle, product, fits_product)
    label_filepath = str(product.label_filepath)
    with open(label_filepath, "w") as f:
        f.write(label)

    # POSTCONDITION
    assert os.path.isfile(str(product.label_filepath))
    # TODO looks like the wrong path

    return label


def _make_product_spice_kernel_label(bundle, product, fits_product):
    # type: (B.Bundle, P.Product, Product) -> str
    """
    Given the Bundle and Product objects and FitsProduct row, create a
    product label and return it.  TODO unimplemented
    """
    # PRECONDITION
    assert False, "ensure this implementation makes sense"  # it doesn't
    # assert spice_kernel exists

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
    files = [('bob', 'PDF')]  # TODO

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
        logging.getLogger(__name__).error(label)
        raise

    # POSTCONDITION
    assert pretty

    return pretty


##############################


PRODUCT_LID = 'urn:nasa:pds:hst_14334:data_wfc3_raw:icwy08q3q_raw'
# type: str


def _run():
    archive = get_any_archive()
    fits_product = P.Product(archive, LID(PRODUCT_LID))
    collection = fits_product.collection()
    bundle = fits_product.bundle()
    db_fp = bundle_database_filepath(bundle)
    print db_fp
    session = create_database_tables_and_session(db_fp)

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
    _run()
