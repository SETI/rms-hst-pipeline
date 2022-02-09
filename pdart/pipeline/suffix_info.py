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
# 1. boolean checking if it's an accepted suffixe that we want to download
# 2. processing_level
# 3. collection_type
# 4. product title
# 5. collection title
# 6. a list of instrument ids
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
    "asc": (  
        True,
        "None",
        "Miscellaneous",
        "Post-calibration {I} association file {F}, describing an observation set in HST Program {P}",
        "Post-calibration {I} association files describing observation sets in HST Program {P}",
        ["NICMOS"],
    ),
    "asn": (  
        True,
        "None",
        "Miscellaneous",
        "{I} association file {F}, describing an observation set in HST Program {P}",
        "{I} association files describing observation sets in HST Program {P}",
        ["ACS", "NICMOS", "STIS", "WFC3"],
    ),
    "cal": (
        True,
        "Calibrated",
        "Data",
        "Calibrated {I} image file {F} from HST Program {P}",
        "Calibrated {I} image files from HST Program {P}",
        ["NICMOS"],
    ),
    "clb": (
        True,
        "Calibration",
        "Data",
        "Calibrated {I} background image file {F} for ACCUM observations in HST Program {P}",
        "Calibrated {I} background image files for ACCUM observations in HST Program {P}",
        ["NICMOS"],
    ),
    "clf": (
        True,
        "Calibration",
        "Data",
        "Calibrated {I} flat-field image file {F} for ACCUM observations in HST Program {P}",
        "Calibrated {I} flat-field image files for ACCUM observations in HST Program {P}",
        ["NICMOS"],
    ),
    "crc": (  # if available, then crj should be redefined below
        True,
        "Calibrated",
        "Data",
        "Combined, calibrated, CTE-corrected {I} image file {F} from repeated exposures in HST Program {P}",
        "Combined, calibrated, CTE-corrected {I} image files from repeated exposures in HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "crj": (  
        True,
        "Calibrated",
        "Data",
        "Combined, calibrated {I} image file {F} from repeated exposures in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures in HST Program {P}",
        ["ACS", "STIS", "WFC3"],
    ),
    "drc": (  # if available, then drz should be redefined below
        True,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion and CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion and CTE, from HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "drz": (  
        True,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion, from HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "epc": (  
        True,
        "None",
        "Miscellaneous",
        "{I} CCD temperature table file {F} for observations in HST Program {P}",
        "{I} CCD temperature table files for observations in HST Program {P}",
        ["NICMOS", "STIS"],
    ),
    "flc": (  # if available, then flt should be redefined below
        True,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded, CTE-corrected {I} image file {F} from HST Program {P}",
        "Calibrated, flat-fielded, CTE-corrected {I} image files from HST Program {P}",
        ["ACS", "WFC3"],
    ),
    "flt": (  
        True,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded {I} image file {F} from HST Program {P}",
        "Calibrated, flat-fielded {I} image files from HST Program {P}",
        ["ACS", "STIS", "WFC3"],
    ),
    "ima": (
        False,
        "Partially Processed",
        "Data",
        "Calibrated but uncombined {I} image file {F} for MULTIACCUM observations in HST Program {P}",
        "Calibrated but uncombined {I} image file for MULTIACCUM observations in HST Program {P}",
        ["NICMOS", "WFC3"],
    ),
    "jif": (  
        True,
        "None",
        "Miscellaneous",
        "{I} 2-D jitter histogram image file {F} for observations in HST Program {P}",
        "{I} 2-D jitter histogram image files for observations in HST Program {P}",
        ["STIS"],
    ),
    "jit": (  
        True,
        "None",
        "Miscellaneous",
        "{I} spacecraft pointing jitter table file {F} for observations in HST Program {P}",
        "{I} spacecraft pointing jitter table files for observations in HST Program {P}",
        ["STIS"],
    ),
    "lrc": (  
        True,
        "None",
        "Miscellaneous",
        "{I} local rate check image file {F} for observations in HST Program {P}",
        "{I} local rate check image files for observations in HST Program {P}",
        ["STIS"],
    ),
    "lsp": (  
        True,
        "None",
        "Miscellaneous",
        "{I} LRC support text header file {F} for observations in HST Program {P}",
        "{I} LRC support text header files for observations in HST Program {P}",
        ["STIS"],
    ),
    "mos": (
        True,
        "Derived",
        "Data",
        "Combined, calibrated {I} image file {F} for dithered observations in HST Program {P}",
        "Combined, calibrated {I} image files for dithered observations in HST Program {P}",
        ["NICMOS"],
    ),
    "pdq": (  
        True,
        "None",
        "Miscellaneous",
        "{I} OPUS processing data quality file {F} for observations in HST Program {P}",
        "{I} OPUS processing data quality files for observations in HST Program {P}",
        ["NICMOS", "STIS"],
    ),
    "raw": (  
        True,
        "Raw",
        "Data",
        "Raw, uncalibrated {I} image file {F} from HST Program {P}",
        "Raw, uncalibrated {I} image files from HST Program {P}",
        ["ACS", "NICMOS", "STIS", "WFC3"],
    ),
    "rwb": (  
        True,
        "None",
        "Miscellaneous",
        "Raw {I} background image file {F} for ACCUM observations in HST Program {P}",
        "Raw {I} background image files for ACCUM observations in HST Program {P}",
        ["NICMOS"],
    ),
    "rwf": (  
        True,
        "None",
        "Miscellaneous",
        "Raw {I} flat-field image file {F} for ACCUM observations in HST Program {P}",
        "Raw {I} flat-field image files for ACCUM observations in HST Program {P}",
        ["NICMOS"],
    ),
    "sx1": (  
        True,
        "Calibrated",
        "Data",
        "1-D extracted {I} spectrum file {F} for summed or CR-rejected observations in HST Program {P}",
        "1-D extracted {I} spectrum files for summed or CR-rejected observations in HST Program {P}",
        ["STIS"],
    ),
    "sx2": (  
        True,
        "Calibrated",
        "Data",
        "2-D direct or spectral {I} image file {F} for summed or CR-rejected observations in HST Program {P}",
        "2-D direct or spectral {I} image files for summed or CR-rejected observations in HST Program {P}",
        ["STIS"],
    ),
    "sfl": (  
        True,
        "Calibrated",
        "Data",
        "Calibrated and summed {I} MAMA image file {F} created from sub-exposures in an observation in HST Program {P}",
        "Calibrated and summed {I} MAMA image files created from sub-exposures in an observation in HST Program {P}",
        ["ACS", "STIS"],
    ),
    "spb": (  
        True,
        "None",
        "Miscellaneous",
        "{I} background SHP and UDL information file {F} for ACCUM observations in HST Program {P}",
        "{I} background SHP and UDL information files for ACCUM observations in HST Program {P}",
        ["NICMOS"],
    ),
    "spf": (  
        True,
        "None",
        "Miscellaneous",
        "{I} flat-field SHP and UDL information file {F} for ACCUM observations in HST Program {P}",
        "{I} flat-field SHP and UDL information files for ACCUM observations in HST Program {P}",
        ["NICMOS"],
    ),
    "spt": (  
        True,
        "None",
        "Miscellaneous",
        "{I} telemetry and engineering file {F}, including target definitions, for HST Program {P}",
        "{I} telemetry and engineering files, including target definitions, for HST Program {P}",
        ["ACS", "NICMOS", "STIS", "WFC3"],
    ),
    "tag": (  
        True,
        "None",
        "Miscellaneous",
        "{I} TIME-TAG event list file {F} for observations in HST Program {P}",
        "{I} TIME-TAG event list files for observations in HST Program {P}",
        ["STIS"],
    ),
    "trl": (  
        True,
        "None",
        "Miscellaneous",
        "{I} trailer file {F}, containing a calibration processing log in HST Program {P}",
        "{I} trailer files, containing calibration processing logs for HST Program {P}",
        ["ACS", "NICMOS", "STIS", "WFC3"],
    ),
    "wav": (  
        True,
        "None",
        "Miscellaneous",
        "{I} associated wavecal exposure file {F} for observations in HST Program {P}",
        "{I} associated wavecal exposure files for observations in HST Program {P}",
        ["STIS"],
    ),
    "wsp": (  
        True,
        "None",
        "Miscellaneous",
        "{I} SPT file {F} for WAV observations in HST Program {P}",
        "{I} SPT files for WAV observations in HST Program {P}",
        ["STIS"],
    ),
    "x1d": (  
        True,
        "Calibrated",
        "Data",
        "1-D extracted {I} spectrum file {F} for observations in HST Program {P}",
        "1-D extracted {I} spectrum files for observations in HST Program {P}",
        ["STIS"],
    ),
    "x2d": (  
        True,
        "Calibrated",
        "Data",
        "2-D direct or spectral {I} image file {F} for observations in HST Program {P}",
        "2-D direct or spectral {I} image files for observations in HST Program {P}",
        ["STIS"],
    ),
    ("ACS", "WFC", "flt"): (  # because flc is also available
        True,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded {I} image file {F}, without CTE correction, from HST Program {P}",
        "Calibrated, flat-fielded {I} image files, without CTE correction, from HST Program {P}",
        ["ACS"],
    ),
    ("ACS", "WFC", "crj"): (  # because crc is also available
        True,
        "Calibrated",
        "Data",
        "Combined, calibrated {I} image file {F} from repeated exposures, without CTE correction, in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures, without CTE correction, in HST Program {P}",
        ["ACS"],
    ),
    ("ACS", "WFC", "drz"): (  # because drc is also available
        True,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion but not CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion but not CTE, from HST Program {P}",
        ["ACS"],
    ),
    ("WFC3", "UVIS", "flt"): (  # because flc is also available
        True,
        "Calibrated",
        "Data",
        "Calibrated, flat-fielded {I} image file {F}, without CTE correction, from HST Program {P}",
        "Calibrated, flat-fielded {I} image files, without CTE correction, from HST Program {P}",
        ["WFC3"],
    ),
    ("WFC3", "UVIS", "crj"): (  # because crc is also available
        True,
        "Calibrated",
        "Data",
        "Combined, calibrated {I} image file {F} from repeated exposures, without CTE correction, in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures, without CTE correction, in HST Program {P}",
        ["WFC3"],
    ),
    ("WFC3", "UVIS", "drz"): (  # because drc is also available
        True,
        "Derived",
        "Data",
        "Calibrated {I} image file {F}, corrected for geometric distortion but not CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion but not CTE, from HST Program {P}",
        ["WFC3"],
    ),
}

# First letter of filenames and the corresponding instrument names
INSTRUMENT_FROM_LETTER_CODE = {
    "i": "WFC3",
    "j": "ACS",
    "l": "COS",
    "n": "NICMOS",
    "o": "STIS",
    "u": "WFPC2",
    "w": "WFPC",
    "x": "FOC",
    "y": "FOS",
    "z": "GHRS",
}

# This is used when we only want to download files with shm, spt, and
# shf suffixes. It's also used to extract Hst_Parameter information.
TARGET_IDENTIFICATION_SUFFIXES = ["shm", "spt", "shf"]

# Get the accepted instruments from SUFFIX_INFO.
def _get_accepted_inst_li():
    suffix_li = []
    for key in SUFFIX_INFO.keys():
        for inst in SUFFIX_INFO[key][5]:
            if inst not in suffix_li:
                suffix_li.append(inst)
    return suffix_li


# Get the accepted letter list based on the accepted instruments.
def _get_accepted_letter_codes():
    accepted_letter_li = []
    accepted_inst_li = _get_accepted_inst_li()
    for letter in INSTRUMENT_FROM_LETTER_CODE.keys():
        if (
            INSTRUMENT_FROM_LETTER_CODE[letter] in accepted_inst_li
            and letter not in accepted_letter_li
        ):
            accepted_letter_li.append(letter)
    return accepted_letter_li


# First letter of filenames
ACCEPTED_LETTER_CODES = "".join(_get_accepted_letter_codes())

# Get the key of SUFFIX_INFO based on the passed in parameters.
def _get_suffix_info_key(instrument_id, channel_id, suffix):
    if instrument_id and type(instrument_id) != str:
        raise AttributeError(
            f"{instrument_id} passed into _get_suffix_info_key " + "is not a string."
        )
    if channel_id and type(channel_id) != str:
        raise AttributeError(
            f"{channel_id} passed into _get_suffix_info_key " + "is not a string."
        )
    if suffix and type(suffix) != str:
        raise AttributeError(
            f"{suffix} passed into _get_suffix_info_key " + "is not a string."
        )
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


# If no instrument_id passed in, get the full list of suffixes from SUFFIX_INFO
# based on the boolean value at idx=0.
# If instrument_id is passed in, get the list of suffixes for the instrument.
def get_suffixes_list(instrument_id=None):
    suffix_li = []
    for key in SUFFIX_INFO.keys():
        if SUFFIX_INFO[key][0]:
            if type(key) is tuple:
                if key[-1] not in suffix_li and (
                    instrument_id is None or instrument_id in SUFFIX_INFO[key][5]
                ):
                    suffix_li.append(key[-1])
            else:
                if key not in suffix_li and (
                    instrument_id is None or instrument_id in SUFFIX_INFO[key][5]
                ):
                    suffix_li.append(key)
    return suffix_li


# Return the suffixes considered raw data, in order of preference.
def get_raw_suffix():
    suffix_li = []
    for key in SUFFIX_INFO.keys():
        if SUFFIX_INFO[key][1] == "Raw":
            if type(key) is tuple:
                if key[-1] not in suffix_li:
                    suffix_li.append(key[-1])
            elif key not in suffix_li:
                suffix_li.append(key)
    return suffix_li


def get_titles_format(instrument_id, channel_id, suffix):
    key = _get_suffix_info_key(instrument_id, channel_id, suffix)
    try:
        titles = SUFFIX_INFO[key][3:5]
    except:
        raise ValueError(f"{key} has no titles in SUFFIX_INFO.")
    return titles


def get_processing_level(suffix, instrument_id=None, channel_id=None):
    key = _get_suffix_info_key(instrument_id, channel_id, suffix)
    try:
        processing_level = SUFFIX_INFO[key][1]
    except:
        raise ValueError(f"{key} has no processing level in SUFFIX_INFO.")
        # TODO: SUFFIX_INFO will be updated later. Might need a default value
        # if suffix doesn't exist in SUFFIX_INFO
        # processing_level = "Raw"
    return processing_level


def get_collection_type(suffix, instrument_id=None, channel_id=None):
    key = _get_suffix_info_key(instrument_id, channel_id, suffix)
    try:
        collection_type = SUFFIX_INFO[key][2].lower()
    except:
        raise ValueError(f"{key} has no collection type in SUFFIX_INFO.")
        # TODO: SUFFIX_INFO will be updated later. Might need a default value
        # if suffix doesn't exist in SUFFIX_INFO
        # collection_type = "data"
    return collection_type
