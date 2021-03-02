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

SUFFIX_TITLES = {
    "asn": (
        "{I} association file {F}, describing an observation set in HST Program {P}",
        "{I} association files describing observation sets in HST Program {P}",
    ),
    "trl": (
        "{I} trailer file {F}, containing a calibration processing log in HST Program {P}",
        "{I} trailer files, containing calibration processing logs for HST Program {P}",
    ),
    "spt": (
        "{I} telemetry and engineering file {F}, including target definitions, for HST Program {P}",
        "{I} telemetry and engineering files, including target definitions, for HST Program {P}",
    ),
    "raw": (
        "Raw, uncalibrated {I} image file {F} from HST Program {P}",
        "Raw, uncalibrated {I} image files from HST Program {P}",
    ),
    "flt": (
        "Calibrated, flat-fielded {I} image file {F} from HST Program {P}",
        "Calibrated, flat-fielded {I} image files from HST Program {P}",
    ),
    "flc": (
        "Calibrated, flat-fielded, CTE-corrected {I} image file {F} from HST Program {P}",
        "Calibrated, flat-fielded, CTE-corrected {I} image files from HST Program {P}",
    ),
    "crj": (
        "Combined, calibrated {I} image file {F} from repeated exposures in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures in HST Program {P}",
    ),
    "crc": (
        "Combined, calibrated, CTE-corrected {I} image file {F} from repeated exposures in HST Program {P}",
        "Combined, calibrated, CTE-corrected {I} image files from repeated exposures in HST Program {P}",
    ),
    "drz": (
        "Calibrated {I} image file {F}, corrected for geometric distortion, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion, from HST Program {P}",
    ),
    "drc": (
        "Calibrated {I} image file {F}, corrected for geometric distortion and CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion and CTE, from HST Program {P}",
    ),
    ("ACS", "WFC", "flt"): (
        "Calibrated, flat-fielded {I} image file {F}, without CTE correction, from HST Program {P}",
        "Calibrated, flat-fielded {I} image files, without CTE correction, from HST Program {P}",
    ),
    ("ACS", "WFC", "crj"): (
        "Combined, calibrated {I} image file {F} from repeated exposures, without CTE correction, in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures, without CTE correction, in HST Program {P}",
    ),
    ("ACS", "WFC", "drz"): (
        "Calibrated {I} image file {F}, corrected for geometric distortion but not CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion but not CTE, from HST Program {P}",
    ),
    ("WFC3", "UVIS", "flt"): (
        "Calibrated, flat-fielded {I} image file {F}, without CTE correction, from HST Program {P}",
        "Calibrated, flat-fielded {I} image files, without CTE correction, from HST Program {P}",
    ),
    ("WFC3", "UVIS", "crj"): (
        "Combined, calibrated {I} image file {F} from repeated exposures, without CTE correction, in HST Program {P}",
        "Combined, calibrated {I} image files from repeated exposures, without CTE correction, in HST Program {P}",
    ),
    ("WFC3", "UVIS", "drz"): (
        "Calibrated {I} image file {F}, corrected for geometric distortion but not CTE, from HST Program {P}",
        "Calibrated {I} image files, corrected for geometric distortion but not CTE, from HST Program {P}",
    ),
    ("WFC3", "IR", "raw"): (
        "Raw, uncalibrated {I} image file {F}, without CR rejection, from HST Program {P}",
        "Raw, uncalibrated {I} image files, without CR rejection, from HST Program {P}",
    ),
    ("WFC3", "IR", "ima"): (
        "Calibrated {I} image file {F}, without CR rejection, from HST Program {P}",
        "Calibrated {I} image files, without CR rejection, from HST Program {P}",
    ),
    ("WFC3", "IR", "flt"): (
        "Calibrated {I} image file {F}, without CR rejection, from HST Program {P}",
        "Calibrated {I} image files, without CR rejection, from HST Program {P}",
    ),
    ("NICMOS", "cal"): (
        "Calibrated {I} image file {F} from HST Program {P}",
        "Calibrated {I} image files from HST Program {P}",
    ),
    ("NICMOS", "ima"): (
        "Calibrated but uncombined {I} image file {F} for MULTIACCUM observations in HST Program {P}",
        "Calibrated but uncombined {I} image file for MULTIACCUM observations in HST Program {P}",
    ),
    ("NICMOS", "mos"): (
        "Combined, calibrated {I} image file {F} for dithered observations in HST Program {P}",
        "Combined, calibrated {I} image files for dithered observations in HST Program {P}",
    ),
}


def get_titles_format(instrument_id, channel_id, suffix):
    if (instrument_id, channel_id, suffix) in SUFFIX_TITLES:
        key = (instrument_id, channel_id, suffix)
    elif (instrument_id, suffix) in SUFFIX_TITLES:
        key = (instrument_id, suffix)
    elif suffix in SUFFIX_TITLES:
        key = suffix
    else:
        raise KeyError(
            f"Titles based on instrument_id: {instrument_id}, "
            + f"channel_id: {channel_id}, suffix: {suffix} "
            + "doesn't exists in SUFFIX_TITLES."
        )

    titles = SUFFIX_TITLES[key]
    return titles
