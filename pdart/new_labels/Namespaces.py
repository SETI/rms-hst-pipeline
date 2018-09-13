from pdart.xml.Pds4Version import HST_SHORT_VERSION, PDS4_SHORT_VERSION

BUNDLE_NAMESPACES = """xmlns="http://pds.nasa.gov/pds4/pds/v1"
xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
"""  # type:str

BROWSE_PRODUCT_NAMESPACES = """xmlns="http://pds.nasa.gov/pds4/pds/v1"
xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.xsd"
""" % (PDS4_SHORT_VERSION,)  # type:str

COLLECTION_NAMESPACES = """xmlns="http://pds.nasa.gov/pds4/pds/v1"
xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
"""  # type: str

DOCUMENT_PRODUCT_NAMESPACES = """xmlns="http://pds.nasa.gov/pds4/pds/v1"
xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1 \
http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.xsd"
""" % (PDS4_SHORT_VERSION,)  # type: str

FITS_PRODUCT_NAMESPACES = """xmlns="http://pds.nasa.gov/pds4/pds/v1"
xmlns:hst="http://pds.nasa.gov/pds4/mission/hst/v1"
xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1 \
https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.xsd \
http://pds.nasa.gov/pds4/mission/hst/v1 \
https://pds.nasa.gov/pds4/mission/hst/v1/PDS4_HST_%s.xsd"
""" % (PDS4_SHORT_VERSION, HST_SHORT_VERSION)  # type: str
