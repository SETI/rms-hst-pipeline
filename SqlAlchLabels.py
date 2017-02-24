from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import cast, TYPE_CHECKING

from pdart.pds4.Archives import get_any_archive
import pdart.pds4.Bundle
from pdart.pds4.LID import LID
from pdart.pds4labels.FileContentsXml import AXIS_NAME_TABLE, BITPIX_TABLE
from pdart.xml.Pds4Version import *
from pdart.xml.Pretty import pretty_print
from pdart.xml.Schema import verify_label_or_raise
from pdart.xml.Templates import combine_nodes_into_fragment, \
    interpret_document_template, interpret_template

from SqlAlchTables import Bundle, Card, Collection, Hdu, lookup_card, Product
from SqlAlch import bundle_database_filepath

if TYPE_CHECKING:
    from typing import Any
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
            'Observation_Area': make_observation_area(product.collection),
            'File_Area_Observational': make_file_area_observational(product)
            }).toxml()
    try:
        pretty = pretty_print(label)
    except:
        print label
        raise
    return pretty


##############################

_array_template = interpret_template(
    """<Array>
<NODE name="offset"/>
<axes>1</axes>
<axis_index_order>Last Index Fastest</axis_index_order>
<NODE name="Element_Array" />
<FRAGMENT name="Axis_Arrays" />
</Array>""")
# type: NodeBuilderTemplate


def make_array(hdu):
    # type: (Hdu) -> NodeBuilder
    offset = cast(unicode, hdu.dat_loc)
    return _array_template({
            'offset': make_offset(offset),
            'Element_Array': make_element_array(hdu),
            'Axis_Arrays': _make_axis_arrays(hdu, 1)
            })

##############################

_array_2d_image_template = interpret_template("""<Array_2D_Image>
<NODE name="offset" />
<axes>2</axes>
<axis_index_order>Last Index Fastest</axis_index_order>
<NODE name="Element_Array" />
<FRAGMENT name="Axis_Arrays" />
</Array_2D_Image>""")


def make_array_2d_image(hdu):
    # type: (Hdu) -> NodeBuilder
    offset = cast(unicode, hdu.dat_loc)
    return _array_2d_image_template({
            'offset': make_offset(offset),
            'Element_Array': make_element_array(hdu),
            'Axis_Arrays': _make_axis_arrays(hdu, 2)
            })

##############################

_axis_array_template = interpret_template(
    """<Axis_Array>
<axis_name><NODE name="axis_name" /></axis_name>
<elements><NODE name="elements"/></elements>
<sequence_number><NODE name="sequence_number"/></sequence_number>
</Axis_Array>"""
)


def make_axis_array(hdu, axis_index):
    # type: (Hdu, int) -> NodeBuilder
    return _axis_array_template({
            'axis_name': AXIS_NAME_TABLE[axis_index],
            'elements': str(lookup_card(hdu, 'NAXIS%d' % axis_index)),
            'sequence_number': str(axis_index)
            })

##############################

# TODO this creates a fragment, not an element.  is the break in
# symmetry proper here?


def _make_axis_arrays(hdu, axes):
    # type (Hdu, axes) -> FragmentBuilder
    return combine_nodes_into_fragment(
        [make_axis_array(hdu, i + 1) for i in range(0, axes)]
        )

##############################

_data_type_template = interpret_template(
    """<data_type><NODE name="text"/></data_type>"""
)


def make_data_type(text):
    # type: (unicode) -> NodeBuilder
    return _data_type_template({
            'text': text
            })

##############################

_element_array_template = interpret_template(
    """<Element_Array><NODE name="data_type"/></Element_Array>"""
)


def make_element_array(hdu):
    # type: (Hdu) -> NodeBuilder
    data_type = BITPIX_TABLE[int(lookup_card(hdu, 'BITPIX'))]
    return _element_array_template({
            'data_type': make_data_type(data_type)
            })

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
    hdu_index = cast(int, hdu.hdu_index)
    local_identifier = 'hdu_%d' % hdu_index
    offset = cast(unicode, hdu.hdr_loc)
    return _header_template({
            'local_identifier': make_local_identifier(local_identifier),
            'offset': make_offset(offset),
            'object_length': make_object_length(hdu.dat_loc - hdu.hdr_loc),
            'parsing_standard_id': make_parsing_standard_id('FITS 3.0'),
            'description': make_description('Global FITS Header')
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


def make_observation_area(collection):
    # type: (Collection) -> NodeBuilder
    bundle = collection.bundle
    return _observation_area_template({
            'Time_Coordinates': make_time_coordinates(),
            'Investigation_Area': make_investigation_area(bundle),
            'Observing_System': make_observing_system(collection),
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
<FRAGMENT name="hdu_content_fragment"/>
    </File_Area_Observational>""")
# type: NodeBuilderTemplate


def make_file_area_observational(product):
    # type: (Product) -> NodeBuilder
    hdu_content_nodes = []
    # type: List[NodeBuilder]
    for hdu in product.hdus:
        hdu_content_nodes.extend(_make_hdu_content_nodes(hdu))
    return _file_area_observational_template({
            'File': make_file(product.fits_filepath),
            'hdu_content_fragment': combine_nodes_into_fragment(
                hdu_content_nodes)
            })


def _make_hdu_content_nodes(hdu):
    # type: (Hdu) -> List[NodeBuilder]
    header_node = make_header(hdu)
    if 0 + hdu.dat_span:
        data_node = _make_hdu_data_node(hdu)
        return [header_node, data_node]
    else:
        return [header_node]


def _make_hdu_data_node(hdu):
    # type: (Hdu) -> NodeBuilder
    axes = int(lookup_card(hdu, 'NAXIS'))
    assert axes in [1, 2], ('unexpected number of axes = %d' % axes)
    if axes == 1:
        return _make_hdu_1d_data_node(hdu)
    elif axes == 2:
        return _make_hdu_2d_data_node(hdu)


def _make_hdu_1d_data_node(hdu):
    # type: (Hdu) -> NodeBuilder
    return make_array(hdu)


def _make_hdu_2d_data_node(hdu):
    # type: (Hdu) -> NodeBuilder
    return make_array_2d_image(hdu)

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


def make_internal_reference(d):
    # type: (Dict[str, Any]) -> NodeBuilder
    return _internal_reference_template(d)

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
    proposal_id = cast(int, bundle.proposal_id)
    text = 'urn:nasa:pds:context:investigation:investigation.hst_%d::1.0' % \
        proposal_id
    internal_ref = {
        'lidvid_reference': make_lidvid_reference(text),
        'reference_type': make_reference_type('data_to_investigation')
        }
    return _investigation_area_template({
            'name': make_name('HST Observing program %d' %
                              proposal_id),
            'type': make_type('Individual Investigation'),
            'Internal_Reference': make_internal_reference(internal_ref),
            })


##############################

_lid_reference_template = interpret_template(
    """<lid_reference><NODE name="text" /></lid_reference>""")
# type: NodeBuilderTemplate


def make_lid_reference(text):
    # type: (unicode) -> NodeBuilder
    return _lid_reference_template({
            'text': text
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
<name><NODE name="name"/></name>
<type><NODE name="type"/></type>
<NODE name="Internal_Reference" />
</Observing_System_Component>"""
)


def make_observing_system_component(hst_or_inst):
    if hst_or_inst == 'hst':
        ty = 'Spacecraft'
        ref_type = 'is_instrument_host'
    else:
        ty = 'Instrument'
        ref_type = 'is_instrument'
    d = {
        # TODO The name is wrong, but it works
        'lidvid_reference': make_lid_reference(
            _hst_or_instrument_lid[hst_or_inst]),
        'reference_type': make_reference_type(ref_type)
        }
    return _observing_system_component_template({
            'name': _hst_or_instrument_name[hst_or_inst],
            'type': ty,
            'Internal_Reference': make_internal_reference(d)
            })
# type: NodeBuilderTemplate

_hst_or_instrument_lid = {
    'hst': 'urn:nasa:pds:context:instrument_host:spacecraft.hst',
    'acs': 'urn:nasa:pds:context:instrument:insthost.acs.acs',
    'wfc3': 'urn:nasa:pds:context:instrument:insthost.acs.wfc3',
    'wfpc2': 'urn:nasa:pds:context:instrument:insthost.acs.wfpc2'
    }
# type: Dict[str, str]

_hst_or_instrument_name = {
    'hst': 'Hubble Space Telescope',
    'acs': 'Advanced Camera for Surveys',
    # 'abbreviation': 'urn:nasa:pds:context:instrument_host:inthost.acs'
    'wfc3': 'Wide Field Camera 3',
    # 'abbreviation': 'wfc3'
    'wfpc2': 'Wide-Field Planetary Camera 2',
    # 'abbreviation': 'wfpc2'
    }
# type: Dict[str, str]

##############################

_observing_system_template = interpret_template(
    """<Observing_System>
    <name><NODE name="name"/></name>
    <FRAGMENT name="Observing_System_Component" />
    </Observing_System>""")
# type: NodeBuilderTemplate


def make_observing_system(collection):
    # type: (Collection) -> NodeBuilder
    inst = cast(str, collection.instrument)
    return _observing_system_template({
            'name': _observing_system_names[inst],
            'Observing_System_Component': combine_nodes_into_fragment([
                    make_observing_system_component('hst'),
                    make_observing_system_component(inst)
                    ])
            })

_observing_system_names = {
    'acs': 'Hubble Space Telescope Advanced Camera for Surveys',
    'wfc3': 'Hubble Space Telescope Wide Field Camera 3',
    'wfpc2': 'Hubble Space Telescope Wide-Field Planetary Camera 2'
}

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


def make_start_date_time(text):
    # type: (unicode) -> NodeBuilder
    return _start_date_time_template({
            'text': text
            })

##############################

_stop_date_time_template = interpret_template(
    """<stop_date_time><NODE name="text"/></stop_date_time>""")
# type: NodeBuilderTemplate


def make_stop_date_time(text):
    # type: (unicode) -> NodeBuilder
    return _stop_date_time_template({
            'text': text
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
    text = '2001-01-01Z'  # TODO
    return _time_coordinates_template({
            'start_date_time': make_start_date_time(text),
            'stop_date_time': make_stop_date_time(text)
            })


##############################

_title_template = interpret_template(
    """<title><NODE name="title" /></title>""")
# type: NodeBuilderTemplate


def make_title(collection):
    # type: (Collection) -> NodeBuilder
    bundle = collection.bundle
    proposal_id = cast(int, bundle.proposal_id)
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
