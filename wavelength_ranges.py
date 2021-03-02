# type: ignore
################################################################################
# wavelength_ranges() returns a list of wavelength ranges to use within the
# Science_Facets of HST product labels.
################################################################################

import re

# This is so we can use abbreviations below
FULL_NAMES = {
    "UV": "Ultraviolet",
    "VIS": "Visible",
    "NIR": "Near Infrared",
}


def wavelength_ranges(instrument_id, detector_ids, filter_name):
    """A list of wavelength ranges, based on HST parameters. Each item in the
    list is one of these strings:
        "'Ultraviolet", "Visible", "Near Infrared", "Infrared"

    Inputs are the values obtained from functions defined in HstParameters.py:
        instrument_id   the string returned by get_instrument_id()
        detector_ids    the list of strings returned by get_detector_ids()
        filter_name     the string returned by get_filter_name()
    """

    # Get the list as abbreviations (a subset of "UV", "VIS", "NIR")
    abbrevs = wavelength_abbrevs(instrument_id, detector_ids, filter_name)

    # Expand the abbreviations
    ranges = [FULL_NAMES[abbrev] for abbrev in abbrevs]

    # Append 'Infrared' if necessary
    if "NIR" in abbrevs:
        return ranges + ["Infrared"]
    else:
        return ranges


# This dictionary contains values that override the filter-based algorithm
# The key is one of:
#   instrument_id
#   (instrument_id, detector_id)
#   (instrument_id, detector_id, filter_name)

EASY_TRANSLATIONS = {
    "NICMOS": ["NIR"],
    "COS": ["UV"],
    "GHRS": ["UV"],
    ("ACS", "SBN"): ["UV"],
    ("HSP", "UV1"): ["UV"],
    ("HSP", "UV2"): ["UV"],
    ("HSP", "VIS"): ["UV", "VIS", "NIR"],
    ("HSP", "POL"): ["UV", "VIS", "NIR"],
    ("HSP", "PMT"): ["UV", "VIS", "NIR"],
    ("STIS", "FUV-MAMA"): ["UV"],
    ("STIS", "NUV-MAMA"): ["UV"],
    ("WFC3", "IR"): ["NIR"],
    ("FOS", "BLUE", "PRISM"): ["UV", "VIS"],
    ("FOS", "AMBER", "PRISM"): ["UV", "VIS", "NIR"],
}


def wavelength_abbrevs(instrument_id, detector_ids, filter_name):
    """A list of abbreviated wavelength ranges: "UV", "VIS", and/or "NIR"."""

    # Handle easy cases, where wavelength range only depends on instrument or
    # instrument/detector. This also handles the special filter "PRISM" on FOS,
    # whose range depends on the detector.
    if instrument_id in EASY_TRANSLATIONS:
        return EASY_TRANSLATIONS[instrument_id]

    results = []
    for det in detector_ids:
        key = (instrument_id, det)
        if key in EASY_TRANSLATIONS:
            results.append(EASY_TRANSLATIONS[key])

        key = (instrument_id, det, filter_name)
        if key in EASY_TRANSLATIONS:
            results.append(EASY_TRANSLATIONS[key])

    if results:
        return ranges_union(results)

    # Return ranges based on filter name(s)
    return ranges_from_filter(filter_name)


# Boundaries between the definitions of UV, VIS, and NIR, with tiny adjustments
# for known filters
UV_MAX = 400
VIS_MIN = 388
VIS_MAX = 702
NIR_MIN = 647

# This regular expression matches the number inside a filter name, e.g.,
# 'F606W'. This is generally the center wavelength of the bandpass in nm.
FILTER_REGEX = re.compile("[FG](|[QR])(\d+)([WMNHLX]|LP)")

# This is a list of filter/grating names whose ranges would not be correctly
# inferred from the name based on the usual algorithm. This is needed for two
# reasons:
#   - the filter name is non-standard, e.g., "CLEAR" or "POL60V"
#   - the filter is wide enough to encompass a range that is outside the one
#     expected based on the center wavelength alone.

FILTER_EXCEPTIONS = {
    "CLEAR": ["UV", "VIS", "NIR"],
    "F130LP": ["UV", "VIS"],
    "F130LP": ["UV", "VIS", "NIR"],
    "F165LP": ["UV", "VIS", "NIR"],
    "F200LP": ["UV", "VIS", "NIR"],
    "F305LP": ["UV", "VIS"],
    "F350LP": ["UV", "VIS", "NIR"],
    "F370LP": ["UV", "VIS"],
    "F372M": ["UV", "VIS"],
    "F380W": ["UV", "VIS"],
    "F410M": ["UV", "VIS"],
    "F430W": ["UV", "VIS"],
    "F435W": ["UV", "VIS"],
    "F600LP": ["VIS", "NIR"],
    "F606W": ["VIS", "NIR"],
    "F606W": ["VIS", "NIR"],
    "F606W": ["VIS", "NIR"],
    "F622W": ["VIS", "NIR"],
    "F622W": ["VIS", "NIR"],
    "F625W": ["VIS", "NIR"],
    "F675W": ["VIS", "NIR"],
    "F675W": ["VIS", "NIR"],
    "F702W": ["VIS", "NIR"],
    "F718M": ["VIS", "NIR"],
    "FQCH4N": ["VIS", "NIR"],
    "FQUVN": ["UV", "VIS"],
    "FR459M": ["UV", "VIS"],
    "G430L": ["UV", "VIS"],
    "G430M": ["UV", "VIS"],
    "G570H": ["VIS", "NIR"],
    "G650L": ["UV", "VIS", "NIR"],
    "G750L": ["VIS", "NIR"],
    "G750M": ["VIS", "NIR"],
    "G780H": ["VIS", "NIR"],
    "G800L": ["VIS", "NIR"],
    "POL0UV": ["UV", "VIS"],
    "POL0V": ["VIS", "NIR"],
    "POL120UV": ["UV", "VIS"],
    "POL120V": ["VIS", "NIR"],
    "POL60UV": ["UV", "VIS"],
    "POL60V": ["VIS", "NIR"],
}


def filter_number(filter_name):
    """The number embedded within a filter name; otherwise, zero."""

    match = FILTER_REGEX.fullmatch(filter_name)
    if not match:
        return 0

    return int(match.group(2))


def ranges_from_one_filter(filter_name):
    """The list of wavelength ranges associated with a single filter name."""

    # Check the list of exceptions first
    if filter_name in FILTER_EXCEPTIONS:
        return FILTER_EXCEPTIONS[filter_name]

    # Otherwise, derive the wavelength ranges from the center wavelength
    # embedded within the filter name.

    # Get the embedded wavelength value
    value = filter_number(filter_name)
    if value == 0:
        raise ValueError("Filter name not handled: " + filter_name)

    # Associate the wavelength with one or more ranges
    results = []
    if value <= UV_MAX:
        results.append("UV")

    if value >= VIS_MIN and value <= VIS_MAX:
        results.append("VIS")

    if value >= NIR_MIN:
        results.append("NIR")

    return results


def ranges_from_filter(filter_name):
    """The list of wavelength ranges associated with a filter name, or else a
    sequence of names concatenated with "+"."""

    names = filter_name.split("+")
    ranges_list = [ranges_from_one_filter(name) for name in names]

    # If there are multiple overlapping filters, the returned list is the
    # mathematical intersection of the individual ranges
    return ranges_intersection(ranges_list)


def ranges_intersection(ranges_list):
    """The intersection of multiple ranges."""

    # This is easy if there's only one item
    if len(ranges_list) == 1:
        return ranges_list[0]

    # Determine the intersection
    combined = set(ranges_list[0])
    for ranges in ranges_list[1:]:
        combined = combined.intersection(set(ranges))

    return sorted_abbrevs(combined)


def ranges_union(ranges_list):
    """The union of multiple ranges."""

    # This is easy if there's only item
    if len(ranges_list) == 1:
        return ranges_list[0]

    # Determine the union
    combined = ranges_list[0]
    for ranges in ranges_list[1:]:
        combined += ranges

    return sorted_abbrevs(combined)  # also removes duplicates


def sorted_abbrevs(ranges):
    """Range abbreviations sorted to increasing wavelength; duplicates removed."""

    sorted = []
    if "UV" in ranges:
        sorted.append("UV")
    if "VIS" in ranges:
        sorted.append("VIS")
    if "NIR" in ranges:
        sorted.append("NIR")

    return sorted
