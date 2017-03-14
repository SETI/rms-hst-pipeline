import pdart.add_pds_tools
import julian

import os.path

from pdart.pds4labels.FileContentsXml import AXIS_NAME_TABLE, BITPIX_TABLE
from pdart.xml.Templates import combine_nodes_into_fragment, \
    interpret_template

from SqlAlchTables import BrowseProduct, Bundle, Collection, Hdu, \
    lookup_card, Product

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, AnyStr, Tuple
    from pdart.xml.Templates import FragBuilder, NodeBuilder, \
        NodeBuilderTemplate

    _NB = NodeBuilder  # an abbreviation used in long signatures

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
    keyword = 'NAXIS%d' % axis_index
    elements = lookup_card(hdu, keyword)
    assert elements, keyword
    return _axis_array_template({
            'axis_name': AXIS_NAME_TABLE[axis_index],
            'elements': str(elements),
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

_context_area_template = interpret_template(
    """<Context_Area>
<NODE name="Time_Coordinates"/>
<NODE name="Investigation_Area"/>
<NODE name="Observing_System"/>
<NODE name="Target_Identification"/>
</Context_Area>"""
    )


def make_context_area(fits_product):
    # type: (Product) -> NodeBuilder
    collection = fits_product.collection
    bundle = collection.bundle
    targname = lookup_card(fits_product.hdus[0], 'TARGNAME')
    return _context_area_template({
            'Time_Coordinates': make_time_coordinates(fits_product),
            'Investigation_Area': make_investigation_area(bundle),
            'Observing_System': make_observing_system(collection),
            'Target_Identification': make_target_identification(targname)
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

_document_template = interpret_template(
    """<Document>
<NODE name="publication_date"/>
<FRAGMENT name="document_edition"/>
</Document>""")
# type: NodeBuilderTemplate


def make_document(publication_date, files):
    # type: (unicode, List[Tuple[AnyStr, AnyStr]]) -> NodeBuilder
    return _document_template({
            'publication_date': make_publication_date(publication_date),
            'document_edition': combine_nodes_into_fragment([
                    make_document_edition('phase2', files)
                    # TODO Is 'phase2' right?  General enough?
                    ])
            })

##############################

_document_edition_template = interpret_template(
    """<Document_Edition>
<NODE name="edition_name"/>
<NODE name="language"/>
<NODE name="files"/>
<FRAGMENT name="document_file"/>
</Document_Edition>""")
# type: NodeBuilderTemplate


def make_document_edition(edition_name, files):
    # type: (AnyStr, List[Tuple[AnyStr, AnyStr]]) -> NodeBuilder
    return _document_edition_template({
            'edition_name': make_edition_name(edition_name),
            'language': make_language('English'),
            'files': make_files(str(len(files))),
            'document_file': combine_nodes_into_fragment([
                    make_document_file(file_name, std)
                    for (file_name, std) in files
                    ])
            })

##############################

_document_file_template = interpret_template(
    """<Document_File>
<NODE name="file_name"/>
<NODE name="document_standard_id"/>
</Document_File>""")
# type: NodeBuilderTemplate


def make_document_file(file_name, document_standard_id):
    # type: (unicode, unicode) -> NodeBuilder
    doc_std_id = make_document_standard_id(document_standard_id)
    return _document_file_template({
            'file_name': make_file_name(file_name),
            'document_standard_id': doc_std_id
            })

##############################

_document_standard_id_template = interpret_template(
    """<document_standard_id><NODE name="text"/></document_standard_id>"""
    )


def make_document_standard_id(text):
    # type: (AnyStr) -> NodeBuilder
    return _document_standard_id_template({
            'text': text
            })

##############################

_edition_name_template = interpret_template(
    """<edition_name><NODE name="text"/></edition_name>"""
    )


def make_edition_name(text):
    # type: (AnyStr) -> NodeBuilder
    return _edition_name_template({
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

_encoded_image_template = interpret_template(
    """<Encoded_Image>
<NODE name="offset"/>
<NODE name="encoding_standard_id" />
</Encoded_Image>"""
    )


def make_encoded_image(product):
    # type: (BrowseProduct) -> NodeBuilder
    object_length = os.path.getsize(cast(unicode, product.browse_filepath))
    return _encoded_image_template({
            'offset': make_offset('0'),
            'object_length': make_object_length(str(object_length)),
            'encoding_standard_id': make_encoding_standard_id('JPEG')
            })

##############################

_encoding_standard_id_template = interpret_template(
    """<encoding_standard_id><NODE name="text"/></encoding_standard_id>"""
    )


def make_encoding_standard_id(text):
    # type: (AnyStr) -> NodeBuilder
    object_length = '0'
    return _encoding_standard_id_template({
            'text': text
            })

##############################

_encoding_type_template = interpret_template(
    """<encoding_type><NODE name="text"/></encoding_type>"""
    )


def make_encoding_type(text):
    # type: (AnyStr) -> NodeBuilder
    object_length = '0'
    return _encoding_type_template({
            'text': text
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

_kernel_type_template = interpret_template(
    """<kernel_type><NODE name="text" /></kernel_type>""")
# type: NodeBuilderTemplate


def make_kernel_type(text):
    # type: (AnyStr) -> NodeBuilder

    return _kernel_type_template({
            'text': text
            })

##############################

_language_template = interpret_template(
    """<language><NODE name="text" /></language>""")
# type: NodeBuilderTemplate


def make_language(text):
    # type: (AnyStr) -> NodeBuilder

    return _language_template({
            'text': text
            })

##############################

_logical_identifier_template = interpret_template(
    """<logical_identifier><NODE name="lid" /></logical_identifier>""")
# type: NodeBuilderTemplate


def make_logical_identifier(text):
    # type: (AnyStr) -> NodeBuilder

    return _logical_identifier_template({
            'lid': text
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

_file_area_browse_template = interpret_template("""<File_Area_Browse>
<NODE name="File"/>
<NODE name="Encoded_Image"/>
</File_Area_Browse>""")
# type: NodeBuilderTemplate


def make_file_area_browse(product):
    # type: (Product) -> NodeBuilder
    return _file_area_browse_template({
            'File': make_file(os.path.basename(product.browse_filepath)),
            'Encoded_Image': make_encoded_image(product)
            })


##############################

_file_area_spice_kernel_template = interpret_template(
    """<File_Area_SPICE_Kernel>
<NODE name="File"/>
<NODE name="SPICE_Kernel"/>
</File_Area_SPICE_Kernel>""")
# type: NodeBuilderTemplate


def make_file_area_spice_kernel(file_basename):
    # type: (unicode) -> NodeBuilder
    return _file_area_spice_kernel_template({
            'File': make_file(file_basename),
            # TODO placeholders below
            'SPICE_Kernel': make_spice_kernel('CK', 'Binary')
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

_files_template = interpret_template(
    """<files><NODE name="text"/></files>""")
# type: NodeBuilderTemplate


def make_files(text):
    # type: (unicode) -> NodeBuilder
    return _files_template({
            'text': text
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

_publication_date_template = interpret_template(
    """<publication_date><NODE name="text"/></publication_date>""")
# type: NodeBuilderTemplate


def make_publication_date(text):
    # type: (unicode) -> NodeBuilder
    return _publication_date_template({
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

_reference_list_template = interpret_template(
    """<Reference_List></Reference_List>""")
# type: NodeBuilderTemplate


def make_reference_list():
    # type: () -> NodeBuilder
    return _reference_list_template({
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

_spice_kernel_template = interpret_template(
    """<SPICE_Kernel>
<NODE name="offset"/>
<NODE name="parsing_standard_id"/>
<NODE name="kernel_type"/>
<NODE name="encoding_type"/>
</SPICE_Kernel>""")
# type: NodeBuilderTemplate


def make_spice_kernel(kernel_type, encoding_type):
    # type: (unicode, unicode) -> NodeBuilder
    return _spice_kernel_template({
            'offset': make_offset('0'),  # TODO Is this right?
            'parsing_standard_id': make_parsing_standard_id('SPICE'),
            'kernel_type': make_kernel_type(kernel_type),
            'encoding_type': make_encoding_type(encoding_type)
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
