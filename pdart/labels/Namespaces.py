from pdart.xml.Pds4Version import HST_SHORT_VERSION, PDS4_SHORT_VERSION

_PDS4_SCHEMA_LOCATION: str = "http://pds.nasa.gov/pds4/pds/v1"

_VERSIONED_PDS4_SCHEMA_LOCATION: str = f"https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_{PDS4_SHORT_VERSION}.xsd"

_VERSIONED_PDS4_SCHEMATRON_LOCATION: str = f"https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_{PDS4_SHORT_VERSION}.sch"


_HST_SCHEMA_LOCATION: str = f"http://pds.nasa.gov/pds4/mission/hst/v1"

_VERSIONED_HST_SCHEMA_LOCATION: str = f"https://pds.nasa.gov/pds4/mission/hst/v1/PDS4_HST_{HST_SHORT_VERSION}.xsd"


_VERSIONED_HST_SCHEMATRON_LOCATION: str = f"https://pds.nasa.gov/pds4/mission/hst/v1/PDS4_HST_{HST_SHORT_VERSION}.sch"


############################################################


_PDS4_NAMESPACE: str = 'xmlns="http://pds.nasa.gov/pds4/pds/v1"'

_HST_NAMESPACE: str = 'xmlns:hst="http://pds.nasa.gov/pds4/mission/hst/v1"'

_PDS_NAMESPACE: str = 'xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"'

_XSI_NAMESPACE: str = 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'


############################################################


def make_namespaces(*parts: str) -> str:
    return " ".join(parts)


def make_schema_locations(*args: str) -> str:
    return f'''xsi:schemaLocation="{' '.join(args)}"'''


############################################################

_NAMESPACES_WITHOUT_HST = make_namespaces(
    _PDS4_NAMESPACE,
    _PDS_NAMESPACE,
    _XSI_NAMESPACE,
    make_schema_locations(_PDS4_SCHEMA_LOCATION, _VERSIONED_PDS4_SCHEMA_LOCATION),
)

_NAMESPACES_WITH_HST = make_namespaces(
    _PDS4_NAMESPACE,
    _PDS_NAMESPACE,
    _HST_NAMESPACE,
    _XSI_NAMESPACE,
    make_schema_locations(
        _PDS4_SCHEMA_LOCATION,
        _VERSIONED_PDS4_SCHEMA_LOCATION,
        _HST_SCHEMA_LOCATION,
        _VERSIONED_HST_SCHEMA_LOCATION,
    ),
)


BUNDLE_NAMESPACES = _NAMESPACES_WITHOUT_HST

BROWSE_PRODUCT_NAMESPACES = _NAMESPACES_WITHOUT_HST

COLLECTION_NAMESPACES = _NAMESPACES_WITHOUT_HST

DOCUMENT_PRODUCT_NAMESPACES = _NAMESPACES_WITHOUT_HST

FITS_PRODUCT_NAMESPACES = _NAMESPACES_WITH_HST


############################################################


def make_xml_model(href: str) -> str:
    return f'<?xml-model href="{href}" \
schematypens="http://purl.oclc.org/dsdl/schematron"?>'


PDS4_XML_MODEL = make_xml_model(_VERSIONED_PDS4_SCHEMATRON_LOCATION)

HST_XML_MODEL = make_xml_model(_VERSIONED_HST_SCHEMATRON_LOCATION)
