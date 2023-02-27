##########################################################################################
# hst_helper/__init__.py
##########################################################################################
import os
from collections import defaultdict

# default start and end date of observation in query mast constraints
START_DATE = (1900, 1, 1)
END_DATE = (2025, 1, 1)

# default number of connection retry when connecting to mast failed.
RETRY = 1

# file directories
HST_DIR = {'staging': os.environ['HST_STAGING'],
           'pipeline': os.environ['HST_PIPELINE'],
           'bundles': os.environ['HST_BUNDLES']}

# suffixes for proposal files
DOCUMENT_EXT = ('apt', 'pdf', 'pro', 'prop')
DOCUMENT_EXT_FOR_CITATION_INFO = ('apt', 'pro')

# File names
PROGRAM_INFO_FILE = 'program-info.txt'
PRODUCTS_FILE = 'products.txt'
TRL_CHECKSUMS_FILE = 'trl_checksums.txt'

# Instrument ids dictionary, keyed by propoposal id and store the list of instrument ids
INST_ID_DICT = defaultdict(set)

# Citation info dictionary, keyed by proposal id nad store the citation info of a
# proposal id
CITATION_INFO_DICT = {}

# Instrument ids dictionary, keyed by propoposal id and store the list of target id info
TARG_ID_DICT = defaultdict(list)

# time coordinates dictionary, keyed by propoposal id and store the roll up start/stop
# time for a collection name of a given proposal id
TIME_DICT = defaultdict(dict)

# TODO: These are for schema csv & label, need to figure how to determine the schema
# inventories and version
PDS4_LIDVID = 'S,urn:nasa:pds:system_bundle:xml_schema:pds-xml_schema::1.19'
HST_LIDVID = 'S,urn:nasa:pds:system_bundle:xml_schema:hst-xml_schema::1.0'
DISP_LIDVID = 'S,urn:nasa:pds:system_bundle:xml_schema:disp-xml_schema::1.17'

# General version of xml label
INFORMATION_MODEL_VERSION = '1.15.0.0'
HST_SHORT_VERSION = '1D00_1000'
DISP_SHORT_VERSION = '1B00'

# Collection names in the bundles
COL_NAME_PREFIX = ['data_', 'miscellaneous_', 'browse_']
