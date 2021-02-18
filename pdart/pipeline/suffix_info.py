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
