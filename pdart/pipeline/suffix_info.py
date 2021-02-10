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
