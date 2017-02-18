from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pdart.pds4.Archives import get_any_archive
import pdart.pds4.Bundle
from pdart.pds4.LID import LID
from pdart.xml.Pds4Version import *
from pdart.xml.Pretty import pretty_print
from pdart.xml.Schema import verify_label_or_raise
from pdart.xml.Templates import combine_nodes_into_fragment, \
    interpret_document_template, interpret_template

from SqlAlchTables import Bundle, Collection, Hdu, Product
from SqlAlch import bundle_database_filepath

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pdart.xml.Templates \
        import DocTemplate, NodeBuilder, NodeBuilderTemplate
    from sqlalchemy.engine import *
    from sqlalchemy.schema import *
    from sqlalchemy.types import *


BUNDLE_LID = 'urn:nasa:pds:hst_14334'
# type: str

PRODUCT_LID = 'urn:nasa:pds:hst_14334:data_wfc3_trl:icwy08q3q_trl'
# type: str


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
    label = _product_observational_template({
            'Identification_Area': make_identification_area(product),
            'Observation_Area': make_observation_area(
                product.collection.bundle),
            'File_Area_Observational': make_file_area_observational(product)
            }).toxml()
    return pretty_print(label)


##############################

_header_template = interpret_template(
    """<Header>
<NODE name="local_identifier"/>
<NODE name="offset"/>
<NODE name="object_length"/>
<NODE name="parsing_standard_id"/>
<NODE name="description"/>
</Header>""")
# type: NodeBuilderTemplate


def make_header(hdu):
    # type: (Hdu) -> NodeBuilder
    return _header_template({
            'local_identifier': make_local_identifier('hdu_0'),  # TODO
            'offset': make_offset('0'),  # TODO
            'object_length': make_object_length('100'),  # TODO
            'parsing_standard_id': make_parsing_standard_id(
                'FITS 3.0'),  # TODO
            'description': make_description('Global FITS Header')  # TODO
            })

##############################

_identification_area_template = interpret_template(
    """<Identification_Area>
<NODE name="logical_identifier" />
<NODE name="version_id" />
<NODE name="title" />
<NODE name="information_model_version" />
<NODE name="product_class" />
</Identification_Area>""")
# type: NodeBuilderTemplate


def make_identification_area(product):
    # type: (Product) -> NodeBuilder
    return _identification_area_template({
            'logical_identifier': make_logical_identifier(product),
            'version_id': make_version_id(),
            'title': make_title(product.collection),
            'information_model_version': make_information_model_version(),
            'product_class': make_product_class(product)
            })


##############################

_logical_identifier_template = interpret_template(
    """<logical_identifier><NODE name="lid" /></logical_identifier>""")
# type: NodeBuilderTemplate


def make_logical_identifier(product):
    # type: (Product) -> NodeBuilder
    return _logical_identifier_template({
            'lid': product.lid
            })

##############################

_name_template = interpret_template(
    """<name><NODE name="text" /></name>""")
# type: NodeBuilderTemplate


def make_name(text):
    # type: (unicode) -> NodeBuilder
    return _name_template({
            'text': text
            })

##############################

_observation_area_template = interpret_template(
    """<Observation_Area>
<NODE name="Time_Coordinates" />
<NODE name="Investigation_Area" />
<NODE name="Observing_System" />
<NODE name="Target_Identification" />
</Observation_Area>""")
# type: NodeBuilderTemplate


def make_observation_area(bundle):
    # type: (Bundle) -> NodeBuilder
    return _observation_area_template({
            'Time_Coordinates': make_time_coordinates(),
            'Investigation_Area': make_investigation_area(bundle),
            'Observing_System': make_observing_system(bundle),
            'Target_Identification': make_target_identification()
            })


##############################

_offset_template = interpret_template(
    """<offset unit="byte"><NODE name="text"/></offset>""")
# type: NodeBuilderTemplate


def make_offset(text):
    # type: (unicode) -> NodeBuilder
    return _offset_template({
            'text': text
            })

##############################

_parsing_standard_id_template = interpret_template(
    """<parsing_standard_id><NODE name="text" /></parsing_standard_id>""")
# type: NodeBuilderTemplate


def make_parsing_standard_id(text):
    # type: (unicode) -> NodeBuilder
    return _parsing_standard_id_template({
            'text': text
            })

##############################

_local_identifier_template = interpret_template(
    """<local_identifier><NODE name="text"/></local_identifier>""")
# type: NodeBuilderTemplate


def make_local_identifier(text):
    # type: (unicode) -> NodeBuilder
    return _local_identifier_template({
            'text': text
            })

##############################

_description_template = interpret_template(
    """<description><NODE name="text"/></description>""")
# type: NodeBuilderTemplate


def make_description(text):
    # type: (unicode) -> NodeBuilder
    return _description_template({
            'text': text
            })

##############################

_object_length_template = interpret_template(
    """<object_length unit="byte"><NODE name="text"/></object_length>""")
# type: NodeBuilderTemplate


def make_object_length(text):
    # type: (unicode) -> NodeBuilder
    return _object_length_template({
            'text': text
            })

##############################

_file_area_observational_template = interpret_template(
    """<File_Area_Observational><NODE name="File" />
    <NODE name="Header"/>
    </File_Area_Observational>""")
# type: NodeBuilderTemplate


def make_file_area_observational(product):
    # type: (Product) -> NodeBuilder
    return _file_area_observational_template({
            'File': make_file('foo_bar.fits'),  # TODO
            'Header': make_header(product.hdus[0])  # TODO
            })


##############################

_file_template = interpret_template(
    """<File><NODE name="file_name"/></File>""")
# type: NodeBuilderTemplate


def make_file(text):
    # type: (unicode) -> NodeBuilder
    return _file_template({
            'file_name': make_file_name(text)
            })

##############################

_file_name_template = interpret_template(
    """<file_name><NODE name="text"/></file_name>""")
# type: NodeBuilderTemplate


def make_file_name(text):
    # type: (unicode) -> NodeBuilder
    return _file_name_template({
            'text': text
            })

##############################

_information_model_version_template = interpret_template(
    """<information_model_version><NODE name="version" />\
</information_model_version>""")
# type: NodeBuilderTemplate


def make_information_model_version():
    # type: () -> NodeBuilder
    return _information_model_version_template({
            'version': '1.6.0.0'  # TODO What is this?
            })

##############################

_internal_reference_template = interpret_template(
    """<Internal_Reference>
    <NODE name="lidvid_reference" />
    <NODE name="reference_type" />
    </Internal_Reference>""")
# type: NodeBuilderTemplate


def make_internal_reference(bundle):
    # type: (Bundle) -> NodeBuilder
    proposal_id = 0  # TODO
    text = 'urn:nasa:pds:context:investigation:investigation.hst_%d::1.0' % \
        proposal_id
    return _internal_reference_template({
            'lidvid_reference': make_lidvid_reference(text),
            'reference_type': make_reference_type('data_to_investigation')
            })

##############################

_investigation_area_template = interpret_template(
    """<Investigation_Area>
    <NODE name="name"/>
    <NODE name="type"/>
    <NODE name="Internal_Reference"/>
    </Investigation_Area>""")
# type: NodeBuilderTemplate


def make_investigation_area(bundle):
    # type: (Bundle) -> NodeBuilder
    proposal_id = 0  # TODO
    return _investigation_area_template({
            'name': make_name('HST Observing program %d' %
                              proposal_id),  # TODO
            'type': make_type('Individual Investigation'),  # TODO
            'Internal_Reference': make_internal_reference(bundle),
            })


##############################

_lidvid_reference_template = interpret_template(
    """<lidvid_reference><NODE name="text" /></lidvid_reference>""")
# type: NodeBuilderTemplate


def make_lidvid_reference(text):
    # type: (unicode) -> NodeBuilder
    return _lidvid_reference_template({
            'text': text
            })

##############################

_observing_system_component_template = interpret_template(
    """<Observing_System_Component>
    <NODE name="name"/><NODE name="type"/>
    </Observing_System_Component>""")
# type: NodeBuilderTemplate


def make_observing_system_component():
    # type: () -> NodeBuilder
    return _observing_system_component_template({
            'name': make_name('Hubble Space Telescope'),  # TODO
            'type': make_type('Spacecraft')  # TODO
            })

##############################

_observing_system_template = interpret_template(
    """<Observing_System>
    <NODE name="name"/>
    <FRAGMENT name="Observing_System_Component" />
    </Observing_System>""")
# type: NodeBuilderTemplate


def make_observing_system(bundle):
    # type: (Bundle) -> NodeBuilder
    # TODO Generalize this
    return _observing_system_template({
            'name': make_name('FOO'),  # TODO
            'Observing_System_Component': combine_nodes_into_fragment([
                    make_observing_system_component(),
                    make_observing_system_component()
                    ])
            })

##############################

_product_class_template = interpret_template(
    """<product_class><NODE name="text"/></product_class>""")
# type: NodeBuilderTemplate


def make_product_class(product):
    # type: (Product) -> NodeBuilder
    product_type = str(product.type)
    if product_type == 'fits_product':
        text = 'Product_Observational'
    else:
        assert False, 'Unimplemented for ' + product_type

    return _product_class_template({
            'text': text
            })

##############################

_reference_type_template = interpret_template(
    """<reference_type><NODE name="text" /></reference_type>""")
# type: NodeBuilderTemplate


def make_reference_type(text):
    # type: (unicode) -> NodeBuilder
    return _reference_type_template({
            'text': text
            })

##############################

_start_date_time_template = interpret_template(
    """<start_date_time><NODE name="text"/></start_date_time>""")
# type: NodeBuilderTemplate


def make_start_date_time():
    # type: () -> NodeBuilder
    return _start_date_time_template({
            'text': '2001-01-01Z'  # TODO
            })

##############################

_stop_date_time_template = interpret_template(
    """<stop_date_time><NODE name="text"/></stop_date_time>""")
# type: NodeBuilderTemplate


def make_stop_date_time():
    # type: () -> NodeBuilder
    return _stop_date_time_template({
            'text': '2001-01-01Z'  # TODO
            })

##############################

_target_identification_template = interpret_template(
    """<Target_Identification>
<NODE name="name"/>
<NODE name="type"/>
</Target_Identification>""")
# type: NodeBuilderTemplate


def make_target_identification():
    # type: () -> NodeBuilder
    return _target_identification_template({
            'name': make_name('Magrathea'),  # TODO
            'type': make_type('Planet'),  # TODO
            })


##############################

_time_coordinates_template = interpret_template(
    """<Time_Coordinates>
<NODE name="start_date_time"/>
<NODE name="stop_date_time"/>
</Time_Coordinates>""")
# type: NodeBuilderTemplate


def make_time_coordinates():
    # type: () -> NodeBuilder
    text = '2001-01-01'  # TODO
    return _time_coordinates_template({
            'start_date_time': make_start_date_time(),
            'stop_date_time': make_stop_date_time()
            })


##############################

_title_template = interpret_template(
    """<title><NODE name="title" /></title>""")
# type: NodeBuilderTemplate


def make_title(collection):
    # type: (Collection) -> NodeBuilder
    bundle = collection.bundle
    # TODO
    proposal_id = 0  # TODO
    title = ('This product contains the %s image obtained by ' +
             'HST Observing Program %d') % (str(collection.suffix).upper(),
                                            proposal_id)
    return _title_template({
            'title': title
            })

##############################

_type_template = interpret_template(
    """<type><NODE name="text" /></type>""")
# type: NodeBuilderTemplate


def make_type(text):
    # type: (unicode) -> NodeBuilder
    return _type_template({
            'text': text
            })

##############################

_version_id_template = interpret_template(
    """<version_id>0.1</version_id>""")
# type: NodeBuilderTemplate


def make_version_id():
    # type: () -> NodeBuilder
    return _version_id_template({
            })

if __name__ == '__main__':
    archive = get_any_archive()
    bundle = pdart.pds4.Bundle.Bundle(archive, LID(BUNDLE_LID))
    db_fp = bundle_database_filepath(bundle)
    print db_fp
    engine = create_engine('sqlite:///' + db_fp)

    Session = sessionmaker(bind=engine)
    session = Session()
    product = session.query(Product).filter_by(lid=PRODUCT_LID).first()

    label = make_product_observational_label(product)
    print label
    verify_label_or_raise(label)
