# type: ignore

# Key is an HST filename suffix
# Value is the tuple (processing_level, collection_type)
SUFFIX_INFO = {
    "asn": ("Raw", "Miscellaneous"),
    "cal": ("Calibrated", "Data"),
    "crc": ("Calibrated", "Data"),
    "crj": ("Calibrated", "Data"),
    "drc": ("Derived", "Data"),
    "drz": ("Derived", "Data"),
    "flc": ("Calibrated", "Data"),
    "flt": ("Calibrated", "Data"),
    "ima": ("Partially Processed", "Data"),
    "mos": ("Derived", "Data"),
    "raw": ("Raw", "Data"),
    "spt": ("Telemetry", "Miscellaneous"),
    "trl": ("Calibrated", "Miscellaneous"),
}

# For every instrument, we download files with these suffixes.
ACCEPTED_SUFFIXES = [
    "A1F",
    "A2F",
    "A3F",
    "ASC",
    "ASN",
    # "C0F", # waivered
    # "C1F", # waivered
    # "C2F", # waivered
    # "C3F", # waivered
    "C0M",
    "C1M",
    "C2M",
    "C3M",
    "CAL",
    "CLB",
    "CLF",
    "CORRTAG",
    "CQF",
    "CRC",
    "CRJ",
    # "D0F", # waivered
    "D0M",
    "DRC",
    "DRZ",
    "FLC",
    "FLT",
    "FLTSUM",
    "MOS",
    "RAW",
    # "SHF", # waivered
    "SHM",
    "SPT",
    "SX2",
    "SXL",
    "X1D",
    "X1DSUM",
    "X2D",
]

# This is used when we only want to download files with shm & spt suffixes.
PART_OF_ACCEPTED_SUFFIXES = [
    "SHM",
    "SPT",
]

# First letter of filenames
ACCEPTED_INSTRUMENTS = "IJLNOUWXYZ"

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


def get_collection_type(suffix):
    try:
        collection_type = SUFFIX_INFO[suffix][1].lower()
    except:
        raise ValueError(f"{suffix} has no collection type in SUFFIX_INFO.")
        # TODO: SUFFIX_INFO will be updated later. Might need a default value
        # if suffix doesn't exist in SUFFIX_INFO
        # collection_type = "data"
    return collection_type


def get_processing_level(suffix):
    try:
        processing_level = SUFFIX_INFO[suffix][0]
    except:
        raise ValueError(f"{suffix} has no processing level in SUFFIX_INFO.")
        # TODO: SUFFIX_INFO will be updated later. Might need a default value
        # if suffix doesn't exist in SUFFIX_INFO
        # processing_level = "Raw"
    return processing_level
