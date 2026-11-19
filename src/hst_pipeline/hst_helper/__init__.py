##########################################################################################
# hst_helper/__init__.py
#
# This file contains gobal variables shared between multiple tasks.
# The module contains different helper functions used in multiple tasks.
##########################################################################################

import os
from collections import defaultdict

from hst_pipeline.product_labels.suffix_info import BROWSE_EXTENSIONS

# default start and end date of observation in query MAST constraints
START_DATE = (1900, 1, 1)
END_DATE = (2025, 1, 1)

# default number of connection retry when connecting to MAST failed.
RETRY = 1

# file directories, taken from the environment. Every variable below is required; see
# _read_hst_dir() for what happens when one is missing.
_REQUIRED_ENV = (
    ('staging',  'HST_STAGING',  'where files downloaded from MAST are staged'),
    ('pipeline', 'HST_PIPELINE', 'where logs, program info, and the task queue database live'),
    ('bundles',  'HST_BUNDLES',  'where the final deliverable bundles are written'),
)


class MissingEnvironmentError(RuntimeError):
    """Raised at import time when a required environment variable is not set."""


def _read_hst_dir():
    """Return the HST_DIR mapping, or report every missing environment variable at once.

    Returns:
        a dict keyed by 'staging', 'pipeline', and 'bundles', holding the directory
        named by the corresponding environment variable.

    Raises:
        MissingEnvironmentError    if any required variable is unset, naming all of the
                                   missing ones rather than only the first.
    """
    hst_dir = {}
    missing = []
    for key, var, purpose in _REQUIRED_ENV:
        value = os.environ.get(var)
        if value:
            hst_dir[key] = value
        else:
            missing.append((var, purpose))

    if missing:
        lines = ['The HST pipeline requires the following environment '
                 f'variable{"s" if len(missing) > 1 else ""} to be set:', '']
        width = max(len(var) for var, _ in missing)
        lines += [f'    {var:<{width}}  {purpose}' for var, purpose in missing]
        lines += ['', 'For example:', '']
        lines += [f'    export {var}=/path/to/{var.split("_", 1)[1].lower()}'
                  for var, _ in missing]
        lines += ['', 'See the README for details. No pipeline module can be '
                  'imported until they are set.']
        raise MissingEnvironmentError('\n'.join(lines))

    return hst_dir


HST_DIR = _read_hst_dir()

# suffixes for proposal files
DOCUMENT_EXT = ('apt', 'pdf', 'pro', 'prop')
DOCUMENT_EXT_FOR_CITATION_INFO = ('apt', 'pro')

# Fallback source of citation info for early programs with no .apt or .pro file: the MAST
# proposal search page, saved in the pipeline directory as <nnnnn>-mast.html. It is not a
# proposal file, so it is not copied into the bundle's document collection.
MAST_PROPOSAL_URL = 'https://archive.stsci.edu/proposal_search.php?id={}&mission=hst'
MAST_PROPOSAL_SUFFIX = '-mast.html'

# File names
PROGRAM_INFO_FILE = 'program-info.txt'
PRODUCTS_FILE = 'products.txt'
TRL_CHECKSUMS_FILE = 'trl_checksums.txt'

# Instrument ids dictionary, keyed by propoposal id and store the list of instrument ids
INST_ID_DICT = defaultdict(set)

# Citation info dictionary, keyed by proposal id nad store the citation info of a
# proposal id
CITATION_INFO_DICT = {}

# Target identifications dictionary, keyed by propoposal id and store the list of target
# id info
TARG_ID_DICT = defaultdict(list)

# Version id dictionary keyed by proposal id and store a internal dictionary keyed by
# file name with the version id as the value.
# TODO: construct this when we determine the vid for all files, for now put 1.0
VID_DICT = defaultdict(dict)

# The following dictionaries are keyed by proposal id and store internal dictionaries
# keyed by collection name with the label data (roll up start/stop time, instrument
# params, primary results, and records as the values.
TIME_DICT = defaultdict(dict)
INST_PARAMS_DICT = defaultdict(dict)
PRIMARY_RES_DICT = defaultdict(dict)
RECORDS_DICT = defaultdict(dict)

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

# MAST download directory name
MAST_DOWNLOAD_DIRNAME = 'mastDownload'

# Suffix of each program directory under HST_BUNDLES: <HST_BUNDLES>/hst_<nnnnn>-deliverable/
DELIVERABLE_SUFFIX = '-deliverable'

# Prefix for files while downloading; stripped after all downloads succeed.
DOWNLOAD_TMP_PREFIX = 'tmp_'

# Browse product files extension
BROWSE_PROD_EXT = list(BROWSE_EXTENSIONS)

# For citation info, need to fake it to something that will pass the XsdValidator.jar
# for now. required pattern for dummy value: 10\\.\\S+/\\S+
DOI = '10.1/2'
