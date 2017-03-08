import pdart.add_pds_tools
import julian
import picmaker

import os.path

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import *

from pdart.pds4.Archives import get_any_archive
from pdart.pds4.HstFilename import *
from pdart.pds4.LID import LID
import pdart.pds4.Product as P
from pdart.pds4labels.FileContentsXml import AXIS_NAME_TABLE, BITPIX_TABLE
from pdart.xml.Pds4Version import *
from pdart.xml.Pretty import pretty_print
from pdart.xml.Schema import verify_label_or_raise
from pdart.xml.Templates import combine_nodes_into_fragment, \
    interpret_document_template, interpret_template

# import SqlAlch
from SqlAlchTables import BrowseProduct, Bundle, Collection, lookup_card, \
   Product

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Dict, Text, Tuple

    from sqlalchemy.orm import Session, sessionmaker

    import pdart.pds4.Bundle as B
    from pdart.xml.Templates \
        import DocTemplate, FragBuilder, NodeBuilder, NodeBuilderTemplate

    from SqlAlchTables import FitsProduct, Hdu

    _NB = NodeBuilder  # an abbreviation for long signatures


_NEW_DATABASE_NAME = 'sqlalch-database.db'
# type: str
# TODO This is cut-and-pasted from SqlAlch.  Refactor and remove.


def bundle_database_filepath(bundle):
    # type: (B.Bundle) -> unicode

    # TODO This is cut-and-pasted from SqlAlch.  Refactor and remove.
    return os.path.join(bundle.absolute_filepath(), _NEW_DATABASE_NAME)


def ensure_directory(dir):
    # type: (Text) -> None
    """Make the directory if it doesn't already exist."""

    # TODO This is cut-and-pasted from
    # pdart.pds4label.BrowseProductImageReduction.  Refactor and
    # remove.
    try:
        os.mkdir(dir)
    except OSError:
        pass
    assert os.path.isdir(dir), dir


PRODUCT_LID = 'urn:nasa:pds:hst_14334:data_wfc3_raw:icwy08q3q_raw'
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

    logical_identifier = make_logical_identifier(product)
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

    logical_identifier = make_logical_identifier(collection)
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
    logical_identifier = make_logical_identifier(bundle)
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
    # type (Hdu, axes) -> FragBuilder
    return combine_nodes_into_fragment(
        [make_axis_array(hdu, i + 1) for i in range(0, axes)]
        )

##############################

_bundle_template = interpret_template(
    """<Bundle><NODE name="bundle_type"/></Bundle>"""
)


def make_bundle(bundle):
    # type: (Bundle) -> NodeBuilder
    return _bundle_template({
            'bundle_type': make_bundle_type(bundle)
            })

##############################

_bundle_member_entry_template = interpret_template(
    """<Bundle_Member_Entry>
<NODE name="lid_reference" />
<NODE name="member_status" />
<NODE name="reference_type" />
</Bundle_Member_Entry>"""
)


def make_bundle_member_entry(collection):
    # type: (Collection) -> NodeBuilder
    return _bundle_member_entry_template({
            'lid_reference': make_lid_reference(unicode(collection.lid)),
            'member_status': make_member_status('Primary'),
            'reference_type': make_reference_type('bundle_has_data_collection')
            })

##############################

_bundle_type_template = interpret_template(
    """<bundle_type><NODE name="bundle_type"/></bundle_type>"""
)


def make_bundle_type(bundle):
    # type: (Bundle) -> NodeBuilder
    return _bundle_type_template({
            'bundle_type': 'Archive'
            })

##############################

_citation_information_template = interpret_template(
    """<Citation_Information>\
<NODE name="publication_year"/>\
<NODE name="description"/>\
</Citation_Information>"""
)


def make_citation_information(publication_year, description):
    # type: (unicode, unicode) -> NodeBuilder
    return _citation_information_template({
            'publication_year': make_publication_year(publication_year),
            'description': make_description(description)
            })

##############################

_collection_template = interpret_template(
    """<Collection><NODE name="collection_type"/></Collection>"""
)


def make_collection(collection):
    # type: (Collection) -> NodeBuilder
    return _collection_template({
            # TODO if this isn't constant, generalize the
            # implementation
            'collection_type': make_collection_type('Data')
            })

##############################

_collection_type_template = interpret_template(
    """<collection_type><NODE name="text"/></collection_type>"""
)


def make_collection_type(text):
    # type: (unicode) -> NodeBuilder
    return _collection_type_template({
            'text': text
            })

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

_description_template = interpret_template(
    """<description><NODE name="text"/></description>""")
# type: NodeBuilderTemplate


def make_description(text):
    # type: (unicode) -> NodeBuilder
    return _description_template({
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
<FRAGMENT name="Citation_Information" />
</Identification_Area>""")
# type: NodeBuilderTemplate


def make_identification_area(logical_identifier,
                             version_id,
                             title,
                             information_model_version,
                             product_class,
                             citation_information_fragment):
    # type: (_NB, _NB, _NB, _NB, _NB, FragBuilder) -> _NB

    # Since make_identification_area() is shared by products and
    # collections, instead of passing a high-level object, we pass
    # NodeBuilders for the XML components of <Identification_Area />.
    return _identification_area_template({
            'logical_identifier': logical_identifier,
            'version_id': version_id,
            'title': title,
            'information_model_version': information_model_version,
            'product_class': product_class,
            'Citation_Information': citation_information_fragment
            })


##############################

_logical_identifier_template = interpret_template(
    """<logical_identifier><NODE name="lid" /></logical_identifier>""")
# type: NodeBuilderTemplate


def make_logical_identifier(product):
    # type: (Product) -> NodeBuilder

    # TODO above type is wrong: it's bundle or collection or product.
    return _logical_identifier_template({
            'lid': product.lid
            })

##############################

_maximum_field_length_template = interpret_template(
    """<maximum_field_length unit="byte">\
<NODE name="text" />\
</maximum_field_length>""")
# type: NodeBuilderTemplate


def make_maximum_field_length(text):
    # type: (unicode) -> NodeBuilder
    return _maximum_field_length_template({
            'text': text
            })

##############################

_member_status_template = interpret_template(
    """<member_status><NODE name="text" /></member_status>""")
# type: NodeBuilderTemplate


def make_member_status(text):
    # type: (unicode) -> NodeBuilder
    return _member_status_template({
            'text': text
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


def make_observation_area(product):
    # type: (Product) -> NodeBuilder
    collection = product.collection
    bundle = collection.bundle
    targname = lookup_card(product.hdus[0], 'TARGNAME')
    return _observation_area_template({
            'Time_Coordinates': make_time_coordinates(product),
            'Investigation_Area': make_investigation_area(bundle),
            'Observing_System': make_observing_system(collection),
            'Target_Identification': make_target_identification(targname)
            })


##############################

_field_delimited_template = interpret_template(
    """<Field_Delimited>
<NODE name="name" />
<NODE name="field_number" />
<NODE name="data_type" />
<NODE name="maximum_field_length" />
</Field_Delimited>""")
# type: NodeBuilderTemplate


def make_field_delimited(name, field_number, data_type, maximum_field_length):
    # type: (_NB, _NB, _NB, _NB) -> NodeBuilder
    return _field_delimited_template({
            'name': name,
            'field_number': field_number,
            'data_type': data_type,
            'maximum_field_length': maximum_field_length,
            })

##############################

_field_delimiter_template = interpret_template(
    """<field_delimiter><NODE name="text" /></field_delimiter>""")
# type: NodeBuilderTemplate


def make_field_delimiter(text):
    # type: (unicode) -> NodeBuilder
    return _field_delimiter_template({
            'text': text
            })

##############################

_field_number_template = interpret_template(
    """<field_number><NODE name="text"/></field_number>""")
# type: NodeBuilderTemplate


def make_field_number(text):
    # type: (unicode) -> NodeBuilder
    return _field_number_template({
            'text': text
            })

##############################

_fields_template = interpret_template(
    """<fields><NODE name="text"/></fields>""")
# type: NodeBuilderTemplate


def make_fields(text):
    # type: (unicode) -> NodeBuilder
    return _fields_template({
            'text': text
            })

##############################

_file_area_inventory_template = interpret_template("""<File_Area_Inventory>
<NODE name="File"/>
<NODE name="Inventory"/>
</File_Area_Inventory>""")
# type: NodeBuilderTemplate


def make_file_area_inventory(collection):
    # type: (Collection) -> NodeBuilder
    return _file_area_inventory_template({
            'File': make_file(cast(unicode, collection.inventory_name)),
            'Inventory': make_inventory(collection)
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

_groups_template = interpret_template(
    """<groups><NODE name="text"/></groups>""")
# type: NodeBuilderTemplate


def make_groups(text):
    # type: (unicode) -> NodeBuilder
    return _groups_template({
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

_inventory_template = interpret_template(
    """<Inventory>
<NODE name="offset"/>
<NODE name="parsing_standard_id"/>
<NODE name="records"/>
<NODE name="record_delimiter"/>
<NODE name="field_delimiter"/>
<NODE name="Record_Delimited"/>
<NODE name="reference_type"/>
</Inventory>""")
# type: NodeBuilderTemplate


def make_inventory(collection):
    # type: (Collection) -> NodeBuilder
    return _inventory_template({
            'offset': make_offset('0'),
            'parsing_standard_id': make_parsing_standard_id('PDS DSV 1'),
            'records': make_records('1'),
            'record_delimiter': make_record_delimiter(
                    'Carriage-Return Line-Feed'),
            'field_delimiter': make_field_delimiter('Comma'),
            'Record_Delimited': make_record_delimited(),
            'reference_type': make_reference_type(
                'inventory_has_member_product')
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

_local_identifier_template = interpret_template(
    """<local_identifier><NODE name="text"/></local_identifier>""")
# type: NodeBuilderTemplate


def make_local_identifier(text):
    # type: (unicode) -> NodeBuilder
    return _local_identifier_template({
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

_product_class_template = interpret_template(
    """<product_class><NODE name="text"/></product_class>""")
# type: NodeBuilderTemplate


def make_product_class(text):
    # type: (unicode) -> NodeBuilder
    return _product_class_template({
            'text': text
            })

##############################

_publication_year_template = interpret_template(
    """<publication_year><NODE name="text"/></publication_year>""")
# type: NodeBuilderTemplate


def make_publication_year(text):
    # type: (unicode) -> NodeBuilder
    return _publication_year_template({
            'text': text
            })

##############################

_record_delimited_template = interpret_template(
    """<Record_Delimited>
<NODE name="fields"/>
<NODE name="groups"/>
<FRAGMENT name="field_delimited"/>
</Record_Delimited>""")
# type: NodeBuilderTemplate


def make_record_delimited():
    # type: () -> NodeBuilder
    fields = [
        make_field_delimited(
            make_name('Member Status'),
            make_field_number('1'),
            make_data_type('ASCII_String'),
            make_maximum_field_length('1')
            ),
        make_field_delimited(
            make_name('LIDVID_LID'),
            make_field_number('2'),
            make_data_type('ASCII_LIDVID_LID'),
            make_maximum_field_length('255')
            )
        ]
    return _record_delimited_template({
            'fields': make_fields(str(len(fields))),
            'groups': make_groups('0'),
            'field_delimited': combine_nodes_into_fragment(fields)
            })

##############################

_record_delimiter_template = interpret_template(
    """<record_delimiter><NODE name="text" /></record_delimiter>""")
# type: NodeBuilderTemplate


def make_record_delimiter(text):
    # type: (unicode) -> NodeBuilder
    return _record_delimiter_template({
            'text': text
            })

##############################

_records_template = interpret_template(
    """<records><NODE name="text" /></records>""")
# type: NodeBuilderTemplate


def make_records(text):
    # type: (unicode) -> NodeBuilder
    return _records_template({
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
<NODE name="description"/>
</Target_Identification>""")
# type: NodeBuilderTemplate


def make_target_identification(targname):
    # type: (unicode) -> NodeBuilder
    for prefix, (name, type) in _approximate_target_table.iteritems():
        if targname.startswith(prefix):
            description = 'The %s %s' % (type.lower(), name)
            return _target_identification_template({
                    'name': make_name(name),
                    'type': make_type(type),
                    'description': make_description(description)
                    })

    # TODO remove dummy catch-all
    return _target_identification_template({
            'name': make_name('Magrathea'),
            'type': make_type('Planet'),
            'description': make_description('The planet Magrathea')
            })


_approximate_target_table = {
    'JUP': ('Jupiter', 'Planet'),
    'SAT': ('Saturn', 'Planet'),
    'URA': ('Uranus', 'Planet'),
    'NEP': ('Neptune', 'Planet'),
    'PLU': ('Pluto', 'Dwarf Planet'),
    'PLCH': ('Pluto', 'Dwarf Planet'),
    'IO': ('Io', 'Satellite'),
    'EUR': ('Europa', 'Satellite'),
    'GAN': ('Ganymede', 'Satellite'),
    'CALL': ('Callisto', 'Satellite'),
    'TITAN': ('Titan', 'Satellite'),
    'TRIT': ('Triton', 'Satellite'),
    'DIONE': ('Dione', 'Satellite'),
    'IAPETUS': ('Iapetus', 'Satellite')
    }
# type: Dict[str, Tuple[unicode, unicode]]


##############################

_time_coordinates_template = interpret_template(
    """<Time_Coordinates>
<NODE name="start_date_time"/>
<NODE name="stop_date_time"/>
</Time_Coordinates>""")
# type: NodeBuilderTemplate


def make_time_coordinates(product):
    # type: (Product) -> NodeBuilder
    # TODO figure out and remove coersions
    date_obs = str(lookup_card(product.hdus[0], 'DATE-OBS'))
    time_obs = str(lookup_card(product.hdus[0], 'TIME-OBS'))
    exptime = float('' + lookup_card(product.hdus[0], 'EXPTIME'))

    start_date_time = '%sT%sZ' % (date_obs, time_obs)
    stop_date_time = julian.tai_from_iso(start_date_time) + exptime
    stop_date_time = julian.iso_from_tai(stop_date_time,
                                         suffix='Z')

    return _time_coordinates_template({
            'start_date_time': make_start_date_time(start_date_time),
            'stop_date_time': make_stop_date_time(stop_date_time)
            })


##############################

_title_template = interpret_template(
    """<title><NODE name="title" /></title>""")
# type: NodeBuilderTemplate


def make_title(text):
    # type: (unicode) -> NodeBuilder
    return _title_template({
            'title': text
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


def make_db_browse_product(session, fits_product, browse_product):
    # type: (Session, P.Product, P.Product) -> BrowseProduct

    lid = str(browse_product.lid)

    # TODO I'm deleting it here, but just for development.
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
    return db_browse_product


if __name__ == '__main__':
    archive = get_any_archive()
    fits_product = P.Product(archive, LID(PRODUCT_LID))
    collection = fits_product.collection()
    bundle = fits_product.bundle()
    db_fp = bundle_database_filepath(bundle)
    print db_fp
    engine = create_engine('sqlite:///' + db_fp)

    session = sessionmaker(bind=engine)()
    # type: Session

    if True:
        db_fits_product = \
            session.query(Product).filter_by(lid=PRODUCT_LID).first()

        label = make_product_observational_label(db_fits_product)
        print label
        verify_label_or_raise(label)

    if True:
        browse_product = fits_product.browse_product()
        # three goals:
        # make browse_product in file system
        make_browse_product(fits_product, browse_product)
        # make browse_product in DB
        make_db_browse_product(session, fits_product, browse_product)
        # make label in file system

    # TODO Build inventory

    if True:
        COLLECTION_LID = str(collection.lid)
        db_collection = \
            session.query(Collection).filter_by(lid=COLLECTION_LID).first()

        label = make_product_collection_label(db_collection)
        print label
        verify_label_or_raise(label)

    if True:
        BUNDLE_LID = str(bundle.lid)
        db_bundle = \
            session.query(Bundle).filter_by(lid=BUNDLE_LID).first()

        label = make_product_bundle_label(db_bundle)
        print label
        verify_label_or_raise(label)
