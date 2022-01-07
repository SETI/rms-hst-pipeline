# type: ignore

# This dictionary is to be used to fill in title strings in the labels of products and collections.
#
# Lookup procedure is as follows:
#   1. If (instrument_id, channel_id, suffix) is a key in the dictionary, use these values
#   2. Otherwise, if (instrument_id, suffix) is a key in the dictionary, use these value
#   3. Otherwise, use the value keyed by the suffix.
#
# Value returned is a tuple of two strings (product_title, collection_title).
#
# Each string is expected to be updated using the format() method, which replaces the embedded
# expressions "{I}", "{F}", and "{P}" with values appropriate to the product or collection.
#
# {I} is to be replaced by instrument/channel, e.g., "ACS/WFC" or "WFC3/IR".
# {F} is to be replaced by the file name. This is only used for product titles, not collection
#     titles.
# {P} is to be replaces by the proposal ID, a four-digit or five-digit number.
#
# Example:
#       product_title.format(I=instrument_id + "/" + channel_id, F=filename, P=proposal_id)
#
# Sources:
#   Tables 2.1 and 2.2 of the ACS Data Handbook
#   Tables 2.1 and 2.2 of the WFC3 Data Handbook
#   Table 2.1 of the NICMOS Data Handbook

# A dictionary that contains suffix info. The dictionary is keyed by: suffix
# (instrument_id, suffix), or (instrument_id, channel_id, suffix). The value is
# a tuple containing the following info:
# 1. boolean checking if it's an accepted suffix
# 2. boolean checking if suffixes are considered as raw data
# 3. boolean checking if suffixes are used to extract Hst_Parameter information
# 4. processing_level
# 5. collection_type
# 6. product title
# 7. collection title
# 8. a list of instrument ids
SUFFIX_INFO = {
    #    "sfl": (               # ACS (add this)
    #    "asc": (               # NICMOS
    #    "clb": (               # NICMOS
    #    "clf": (               # NICMOS
    #    "epc": (               # NICMOS (not needed; we don't archive telemetry)
    #    "rwb": (               # NICMOS (is this raw calibration file? maybe exclude)
    #    "rwf": (               # NICMOS (is this raw calibration file? maybe exclude)
    #    "spb": (               # NICMOS
    #    "spf": (               # NICMOS
    #    "pdq": (               # NICMOS
    "asn": (  # ACS, WCF3, NICMOS
        True,
        False,
        False,
        "Raw",
        "Miscellaneous",
        "{I} association file {F}, describing an observation set in HST Program {P}",
        "{I} association files describing observation sets in HST Program {P}",
        ["ACS", "WFC3", "NICMOS"],
    ),
    "trl": (  # ACS, WCF3, NICMOS
        False,
        False,
        False,
        "Calibrated",
        "Miscellaneous",
        "{I} trailer file {F}, containing a calibration processing log in HST Program {P}",
        "{I} trailer files, containing calibration processing logs for HST Program {P}",
        ["ACS", "WFC3", "NICMOS"],
    ),
    "spt": (  # ACS, WCF3, NICMOS
        True,
        False,
        True,
        "Telemetry",
        "Miscellaneous",
        "{I} telemetry and engineering file {F}, including target definitions, for HST Program {P}",
        "{I} telemetry and engineering files, including target definitions, for HST Program {P}",
        ["ACS", "WFC3", "NICMOS"],
    ),
    "raw": (  # ACS, WCF3, NICMOS
        True,
        True,
        False,
        "Raw",
        "Data",
        "Raw, uncalibrated {I} image file {F} from HST Program {P}",
        "Raw, uncalibrated {I} image files from HST Program {P}",
        ["ACS", "WFC3", "NICMOS"],
    ),
    "flt": (  # ACS, WCF3
        True,
        True,
        False,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded {I} image file {F} from HST Program {P}",
        "Calibrated, flat-fielded {I} image files from HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "flc": (  # ACS, WCF3 (if available, then flt should be redefined below)
        True,
        False,
        False,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded, CTE-corrected {I} image file {F} from HST Program {P}",
        "Calibrated, flat-fielded, CTE-corrected {I} image files from HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "crj": (  # ACS, WCF3
        True,
        True,
        False,
        "Calibrated",
        "Data",
        "Combined, calibrated {I} image file {F} from repeated exposures in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures in HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "crc": (  # ACS, WCF3 (if available, then crj should be redefined below)
        True,
        False,
        False,
        "Calibrated",
        "Data",
        "Combined, calibrated, CTE-corrected {I} image file {F} from repeated exposures in HST Program {P}",
        "Combined, calibrated, CTE-corrected {I} image files from repeated exposures in HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "drz": (  # ACS, WCF3
        True,
        True,
        False,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion, from HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "drc": (  # ACS, WCF3 (if available, then drz should be redefined below)
        True,
        False,
        False,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion and CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion and CTE, from HST Program {P}",
        ["ACS", "WFC3"],
    ),
    ("ACS", "WFC", "flt"): (  # because flc is also available
        True,
        True,
        False,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded {I} image file {F}, without CTE correction, from HST Program {P}",
        "Calibrated, flat-fielded {I} image files, without CTE correction, from HST Program {P}",
        ["ACS"],
    ),
    ("ACS", "WFC", "crj"): (  # because crc is also available
        True,
        True,
        False,
        "Calibrated",
        "Data",
        "Combined, calibrated {I} image file {F} from repeated exposures, without CTE correction, in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures, without CTE correction, in HST Program {P}",
        ["ACS"],
    ),
    ("ACS", "WFC", "drz"): (  # because drc is also available
        True,
        True,
        False,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion but not CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion but not CTE, from HST Program {P}",
        ["ACS"],
    ),
    ("WFC3", "UVIS", "flt"): (  # because flc is also available
        True,
        True,
        False,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded {I} image file {F}, without CTE correction, from HST Program {P}",
        "Calibrated, flat-fielded {I} image files, without CTE correction, from HST Program {P}",
        ["WFC3"],
    ),
    ("WFC3", "UVIS", "crj"): (  # because crc is also available
        True,
        True,
        False,
        "Calibrated",
        "Data",
        "Combined, calibrated {I} image file {F} from repeated exposures, without CTE correction, in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures, without CTE correction, in HST Program {P}",
        ["WFC3"],
    ),
    ("WFC3", "UVIS", "drz"): (  # because drc is also available
        True,
        True,
        False,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion but not CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion but not CTE, from HST Program {P}",
        ["WFC3"],
    ),
    ("WFC3", "IR", "raw"): (  # Why do this?  Possibly remove.
        True,
        True,
        False,
        "Raw",
        "Data",
        "Raw, uncalibrated {I} image file {F}, without CR rejection, from HST Program {P}",
        "Raw, uncalibrated {I} image files, without CR rejection, from HST Program {P}",
        ["WFC3"],
    ),
    ("WFC3", "IR", "ima"): (  # Doesn't match handbook.
        False,
        False,
        False,
        "Partially Processed",
        "Data",
        "Calibrated {I} image file {F}, without CR rejection, from HST Program {P}",
        "Calibrated {I} image files, without CR rejection, from HST Program {P}",
        ["WFC3"],
    ),
    ("WFC3", "IR", "flt"): (  # Why do this?  Possibly remove.
        True,
        True,
        False,
        "Calibrated",
        "Data",
        "Calibrated {I} image file {F}, without CR rejection, from HST Program {P}",
        "Calibrated {I} image files, without CR rejection, from HST Program {P}",
        ["WFC3"],
    ),
    ("NICMOS", "cal"): (
        True,
        False,
        False,
        "Calibrated",
        "Data",
        "Calibrated {I} image file {F} from HST Program {P}",
        "Calibrated {I} image files from HST Program {P}",
        ["NICMOS"],
    ),
    ("NICMOS", "ima"): (
        False,
        False,
        False,
        "Partially Processed",
        "Data",
        "Calibrated but uncombined {I} image file {F} for MULTIACCUM observations in HST Program {P}",
        "Calibrated but uncombined {I} image file for MULTIACCUM observations in HST Program {P}",
        ["NICMOS"],
    ),
    ("NICMOS", "mos"): (
        True,
        False,
        False,
        "Derived",
        "Data",
        "Combined, calibrated {I} image file {F} for dithered observations in HST Program {P}",
        "Combined, calibrated {I} image files for dithered observations in HST Program {P}",
        ["NICMOS"],
    ),
}

# First letter of filenames and the corresponding instrument names
INSTRUMENTS_INFO = {
    "i": "wfc3",
    "j": "acs",
    "l": "cos",
    "n": "nicmos",
    "o": "stis",
    "u": "wfpc2",
    "w": "wfpc",
    "x": "foc",
    "y": "fos",
    "z": "ghrs",
}

# First letter of filenames
ACCEPTED_INSTRUMENTS = "".join(INSTRUMENTS_INFO.keys()).upper()

# This is used when we only want to download files with shm & spt suffixes.
PART_OF_ACCEPTED_SUFFIXES = [
    "SHM",
    "SPT",
]


# Get the list of suffixes from SUFFIX_INFO based on the boolean values of
# each key entry. idx = 0 for accepted suffixes, idx = 1 for raw suffixes,
# idx = 2 for shm suffixes
def _get_suffixes_list(idx):
    suffix_li = []
    for key in SUFFIX_INFO.keys():
        if SUFFIX_INFO[key][idx] == True:
            if type(key) is tuple:
                if key[-1] not in suffix_li:
                    suffix_li.append(key[-1])
            else:
                if key not in suffix_li:
                    suffix_li.append(key)
    return suffix_li


# For every instrument, we download files with these suffixes.
# The concatenated list will be removed once SUFFIX_INFO is fully updated.
ACCEPTED_SUFFIXES = [suffix.upper() for suffix in _get_suffixes_list(0)] + [
    "A1F",
    "A2F",
    "A3F",
    "ASC",
    "C0M",
    "C1M",
    "C2M",
    "C3M",
    "CLB",
    "CLF",
    "CORRTAG",
    "CQF",
    "D0M",
    "FLTSUM",
    "SHM",
    "SX2",
    "SXL",
    "X1D",
    "X1DSUM",
    "X2D",
]

# The suffixes considered raw data, in order of preference.
# ["raw", "flt", "drz", "crj", "d0m", "c0m",]
# The concatenated list will be removed once SUFFIX_INFO is fully updated.
RAW_SUFFIXES = _get_suffixes_list(1) + [
    "d0m",
    "c0m",
]


# The suffixes used to extract Hst_Parameter information.
# ["shm", "spt", "shf"]
# The concatenated list will be removed once SUFFIX_INFO is fully updated.
SHM_SUFFIXES = _get_suffixes_list(2) + ["shm", "shf"]


# For each instrument, only download files with selected suffixes.
# Use ACCEPTED_SUFFIXES for all instruments for now.
INTRUMENT_SELECTED_SUFFIXES = {
    "wfc3": ACCEPTED_SUFFIXES,
    "acs": ACCEPTED_SUFFIXES,
    "cos": ACCEPTED_SUFFIXES,
    "nicmos": ACCEPTED_SUFFIXES,
    "stis": ACCEPTED_SUFFIXES,
    "wfpc2": ACCEPTED_SUFFIXES,
    "wfpc": ACCEPTED_SUFFIXES,
    "foc": ACCEPTED_SUFFIXES,
    "fos": ACCEPTED_SUFFIXES,
    "ghrs": ACCEPTED_SUFFIXES,
}


def _get_suffix_info_key(instrument_id, channel_id, suffix):
    if instrument_id:
        instrument_id = instrument_id.upper()
    if channel_id:
        channel_id = channel_id.upper()
    if suffix:
        suffix = suffix.lower()
    if (instrument_id, channel_id, suffix) in SUFFIX_INFO:
        key = (instrument_id, channel_id, suffix)
    elif (instrument_id, suffix) in SUFFIX_INFO:
        key = (instrument_id, suffix)
    elif suffix in SUFFIX_INFO:
        key = suffix
    else:
        raise KeyError(
            f"Key based on instrument_id: {instrument_id}, "
            + f"channel_id: {channel_id}, suffix: {suffix} "
            + "doesn't exists in SUFFIX_INFO."
        )
    return key


def get_titles_format(instrument_id, channel_id, suffix):
    key = _get_suffix_info_key(instrument_id, channel_id, suffix)
    try:
        titles = SUFFIX_INFO[key][5:7]
    except:
        raise ValueError(f"{key} has no titles in SUFFIX_INFO.")
    return titles


def get_processing_level(suffix, instrument_id=None, channel_id=None):
    key = _get_suffix_info_key(instrument_id, channel_id, suffix)
    try:
        processing_level = SUFFIX_INFO[key][3]
    except:
        raise ValueError(f"{key} has no processing level in SUFFIX_INFO.")
        # TODO: SUFFIX_INFO will be updated later. Might need a default value
        # if suffix doesn't exist in SUFFIX_INFO
        # processing_level = "Raw"
    return processing_level


def get_collection_type(suffix, instrument_id=None, channel_id=None):
    key = _get_suffix_info_key(instrument_id, channel_id, suffix)
    try:
        collection_type = SUFFIX_INFO[key][4].lower()
    except:
        raise ValueError(f"{key} has no collection type in SUFFIX_INFO.")
        # TODO: SUFFIX_INFO will be updated later. Might need a default value
        # if suffix doesn't exist in SUFFIX_INFO
        # collection_type = "data"
    return collection_type
