"""
Functionality to build an ``<hst:HST />`` XML element using a SQLite
database.
"""
from typing import Any, Dict, List

from pdart.labels.HstParametersXml import (
    # get_targeted_detector_id,
    hst,
    # parameters_acs, parameters_general, parameters_wfc3,
    # parameters_wfpc2,
)
from pdart.labels.Lookup import Lookup
from pdart.xml.Templates import NodeBuilder

# All functions have the same input arguments:
#   data_lookups: List[Lookup]
#           a list of all the FITS headers in a data file (raw, d0f, drz, etc.)
#   shf_lookup: Lookup
#           the first fits header of the associated _shf.fits file. (Sometimes
#           named _shm.fits).
# The second argument is needed because sometimes the data file does not contain
# all the info we need.
# Internal function used for error messages
def fname(lookup: Lookup) -> str:
    """
    Not used as an attribute but needed for error messages
    """
    try:
        return lookup["FILENAME"].strip()
    except KeyError:
        pass
    # GHRS, at least, does not contain FILENAME, but ROOTNAME is good enough
    return lookup["ROOTNAME"].strip().lower() + "x_xxx.fits"


##############################
# get_aperture_name
##############################
def get_aperture_name(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<aperture_name />`` XML element.
    """
    instrument = get_instrument_id(data_lookups, shf_lookup)
    if instrument in ("WF/PC", "WFPC2", "HSP"):
        return shf_lookup["APER_1"].strip()
    if instrument == "FOS":
        return shf_lookup["APER_ID"]
    # This is valid for most instruments
    try:
        return data_lookups[0]["APERTURE"].strip()
    except KeyError:
        pass
    raise ValueError("missing aperture for " + fname(shf_lookup))


##############################
# get_bandwidth
##############################
def get_bandwidth(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return a float for the ``<bandwidth />`` XML element.
    """
    lookup = data_lookups[0]
    # Works for STIS and WFPC2
    try:
        return "%.4f" % (float(lookup["BANDWID"]) * 1.0e-4)
    except KeyError:
        return "0."


##############################
# get_binning_mode
##############################
def get_binning_mode(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<binning_mode />`` XML element.
    """
    lookup = data_lookups[0]
    instrument = get_instrument_id(data_lookups, shf_lookup)
    # WF/PC and WFPC2 are special cases
    if instrument in ("WF/PC", "WFPC2"):
        obsmode = lookup["MODE"].strip()
        if obsmode == "FULL":
            return "1"
        else:  # obsmode == "AREA"
            return "2"
    # Binning info can be in the first or second FITS header
    for lookup in data_lookups[:2]:
        try:
            binaxis1 = lookup["BINAXIS1"]
            binaxis2 = lookup["BINAXIS2"]
            return str(max(binaxis1, binaxis2))
        except KeyError:
            pass
    return "1"


##############################
# get_center_filter_wavelength
##############################
def get_center_filter_wavelength(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return a float for the ``<center_filter_wavelength />`` XML element.
    """
    lookup = data_lookups[0]
    # Works for STIS and WFPC2
    try:
        return "%.4f" % (float(lookup["CENTRWV"]) * 1.0e-4)
    except KeyError:
        return "0."


##############################
# get_channel_id
##############################
def get_channel_id(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<channel_id />`` XML element.
    """
    lookup = data_lookups[0]
    instrument = get_instrument_id(data_lookups, shf_lookup)
    if instrument == "NICMOS":
        return "NIC" + str(lookup["CAMERA"])
    if instrument == "WF/PC":
        return lookup["CAMERA"].strip()
    # Default behavior
    try:
        return lookup["DETECTOR"].strip()
    except KeyError:
        return instrument


##############################
# get_coronagraph_flag
##############################
def get_coronagraph_flag(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<coronagraph_flag />`` XML element.
    """
    instrument = get_instrument_id(data_lookups, shf_lookup)
    aperture = get_aperture_name(data_lookups, shf_lookup)
    if instrument == "ACS":
        if aperture.startswith("HRC-CORON") or aperture.startswith("HRC-OCCULT"):
            return "true"
    if instrument == "STIS":
        if (
            aperture == "50CORON"
            or aperture.startswith("BAR")
            or aperture.startswith("WEDGE")
            or aperture.startswith("52X0.2F1")
        ):
            return "true"
    if instrument == "NICMOS":
        if aperture == "NIC2-CORON":
            return "true"
    return "false"


##############################
# get_cosmic_ray_split_count
##############################
def get_cosmic_ray_split_count(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<cosmic_ray_split_count />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        return str(lookup["CRSPLIT"])
    except KeyError:
        return "1"  # no CR-splitting unless explicitly stated


##############################
# get_detector_ids
##############################
WFPC2_DETECTOR_IDS = {1: "PC1", 2: "WF2", 3: "WF3", 4: "WF4"}


def get_detector_ids(data_lookups: List[Lookup], shf_lookup: Lookup) -> List[str]:
    """
    Return a list of zero or more text values for the ``<detector_id />``
    XML elements.
    """
    # Interior function
    def get_ccds_from_lookups(data_lookups: List[Lookup], fitsname: str) -> List[int]:
        ccds = []
        for lookup in data_lookups:
            try:
                ccdchip = int(lookup[fitsname])  # weirdly, sometimes a string
                ccds.append(ccdchip)
            except KeyError:
                pass
        ccds = list(set(ccds))  # select unique values
        ccds.sort()
        return ccds

    lookup = data_lookups[0]
    instrument = get_instrument_id(data_lookups, shf_lookup)
    channel = get_channel_id(data_lookups, shf_lookup)
    if instrument == "ACS" and channel == "WFC":
        ccds = get_ccds_from_lookups(data_lookups, "CCDCHIP")
        if -999 in ccds:
            ccds = [1, 2]
        return [f"WFC{k}" for k in ccds]
    if instrument == "COS" and channel == "FUV":
        segment = lookup["SEGMENT"].strip()
        if segment not in ("FUVA", "FUVB", "BOTH"):
            raise ValueError(
                "unrecognized segment (%s) in %s" % (segment, fname(lookup))
            )
        if segment == "FUVA":
            return ["FUVA"]
        elif segment == "FUVB":
            return ["FUVB"]
        else:
            return ["FUVA", "FUVB"]
    if instrument == "GHRS":
        return ["GHRS" + str(lookup["DETECTOR"])]
    if instrument == "HSP":
        config = shf_lookup["CONFIG"].strip()
        # Example: config = HSP/UNK/VIS
        parts = config.split("/")
        assert parts[0] == "HSP", "invalid CONFIG value in " + fname(lookup)
        return [p for p in parts[1:] if p != "UNK"]
    if instrument == "WFC3" and channel == "UVIS":
        ccds = get_ccds_from_lookups(data_lookups, "CCDCHIP")
        if -999 in ccds:
            ccds = [1, 2]
        return [f"UVIS{k}" for k in ccds]
    if instrument == "WF/PC":
        # We will need to find a workaround to read the FITS table from the data
        # file, because that is the only way to get the actual set of detectors
        # if there are less than four! I hope it just doesn't come up.
        count = lookup["NAXIS3"]
        if count != 4:
            raise ValueError(
                "unknown detector subset in (%d/4) in %s" % (count, fname(lookup))
            )
        assert channel in ("PC", "WFC"), "bad channel for " + fname(lookup)
        if channel == "WFC":
            return ["WF1", "WF2", "WF3", "WF4"]
        else:
            return ["PC5", "PC6", "PC7", "PC8"]
    if instrument == "WFPC2":
        ccds = get_ccds_from_lookups(data_lookups, "DETECTOR")
        return [WFPC2_DETECTOR_IDS[k] for k in ccds]
    # Otherwise, return the single value of channel_id
    return [channel]


##############################
# get_exposure_duration
##############################
def get_exposure_duration(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return a float for the ``<exposure_duration />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        return str(lookup["EXPTIME"])
    except KeyError:
        return str(lookup["TEXPTIME"])


##############################
# get_exposure_type
##############################
def get_exposure_type(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<exposure_type />`` XML element.
    """
    lookup = data_lookups[0]
    return lookup["EXPFLAG"].strip()


##############################
# get_filter_name
##############################
def get_filter_name(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<filter_name />`` XML element.
    """
    lookup = data_lookups[0]
    instrument = get_instrument_id(data_lookups, shf_lookup)
    if instrument == "ACS":
        filter1 = lookup["FILTER1"].strip()
        filter2 = lookup["FILTER2"].strip()
        if filter1.startswith("CLEAR"):
            if filter2.startswith("CLEAR"):
                return "CLEAR"
            else:
                return filter1
        if filter2.startswith("CLEAR"):
            return filter1
        # At this point, both filters start with "F" followed by three digits,
        # or "POL" for polarizers. Sort by increasing wavelength; put
        # polarizers second; join with a plus.
        filters = [filter1, filter2]
        filters.sort()
        return "+".join(filters)
    if instrument == "FOC":
        filters = [
            lookup["FILTER1"].strip(),
            lookup["FILTER2"].strip(),
            lookup["FILTER3"].strip(),
            lookup["FILTER4"].strip(),
        ]
        filters = [f for f in filters if not f.startswith("CLEAR")]
        filters.sort()
        return "+".join(filters)
    if instrument in ("FOS", "HSP"):
        return shf_lookup["SPEC_1"].strip()
    if instrument == "GHRS":
        return lookup["GRATING"].strip()
    if instrument == "STIS":
        opt_elem = lookup["OPT_ELEM"].strip()
        parts = lookup["PHOTMODE"].strip().split()
        assert parts[0] == "STIS", "PHOTMODE parsing failure, " + fname(lookup)
        filter = parts.split()[1]
        return opt_elem + "+" + filter
    if instrument in ("WF/PC", "WFPC2"):
        filtnam1 = lookup["FILTNAM1"].strip()
        filtnam2 = lookup["FILTNAM2"].strip()
        if filtnam1 == "":
            return filtnam2
        if filtnam2 == "":
            return filtnam1
        # At this point, both filters start with "F", followed by three digits.
        # Put lower value first; join with a plus.
        filters = [filtnam1, filtnam2]
        filters.sort()
        return "+".join(filters)
    # For other instruments there is just zero or one filter
    try:
        return lookup["FILTER"].strip()
    except KeyError:
        return "Not applicable"


##############################
# get_fine_guidance_sensor_lock_type
##############################
def get_fine_guidance_sensor_lock_type(
    data_lookups: List[Lookup], shf_lookup: Lookup
) -> str:
    """
    Return text for the ``<fine_guidance_system_lock_type />`` XML element.
    """
    lookup = data_lookups[0]
    return lookup["FGSLOCK"].strip()


##############################
# get_gain_setting
##############################
def get_gain_setting(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<gain_mode_id />`` XML element.
    """
    lookup = data_lookups[0]
    # Works for WFPC2
    try:
        gain = int(lookup["ATODGAIN"])  # format WFPC2 gains as ints
        if gain in (7, 15):
            return str(gain)
        raise ValueError("unrecognized WFPC2 gain (%d) in %s" % (gain, fname(lookup)))
    except KeyError:
        pass
    # Works for ACS, WFC3, others
    try:
        gain = lookup["CCDGAIN"]
        return "%3.1f" % gain  # format other gains with one decimal
    except KeyError:
        pass
    return "0."


##############################
# get_gyroscope_mode
##############################
def get_gyroscope_mode(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<gyroscope_mode />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        return str(lookup["GYROMODE"]).strip().replace("T", "3")
    except KeyError:
        return "3"  # Three-gyro mode unless otherwise specified


##############################
# get_hst_pi_name
##############################
def get_hst_pi_name(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<hst_pi_name />`` XML element.
    """
    # Usually in the first FITS header, but in the shf header for GHRS
    for lookup in (data_lookups[0], shf_lookup):
        try:
            pr_inv_l = lookup["PR_INV_L"].strip()
            pr_inv_f = lookup["PR_INV_F"].strip()
            try:
                pr_inv_m = lookup["PR_INV_M"].strip()
            except KeyError:
                pr_inv_m = ""
            return f"{pr_inv_l}, {pr_inv_f} {pr_inv_m}".strip()
        except KeyError:
            pass
    raise ValueError("missing PR_INV_L in " + fname(data_lookups[0]))


##############################
# get_hst_proposal_id
##############################
def get_hst_proposal_id(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<hst_proposal_id />`` XML element.
    """
    lookup = data_lookups[0]
    return str(lookup["PROPOSID"])


##############################
# get_hst_target_name
##############################
def get_hst_target_name(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<hst_target_name />`` XML element.
    """
    lookup = data_lookups[0]
    return lookup["TARGNAME"]


##############################
# get_instrument_id
##############################
def get_instrument_id(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<instrument_id />`` XML element.
    """
    lookup = data_lookups[0]
    instrument = lookup["INSTRUME"].strip()
    if instrument == "HRS":
        return "GHRS"
    if instrument == "WFPC":
        return "WF/PC"
    return instrument


##############################
# get_instrument_mode_id
##############################
def get_instrument_mode_id(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<instrument_mode_id />`` XML element.
    """
    lookup = data_lookups[0]
    instrument = get_instrument_id(data_lookups, shf_lookup)
    if instrument in ("WF/PC", "WFPC2"):
        return lookup["MODE"].strip()
    if instrument == "FOC":
        return lookup["OPTCRLY"].strip()
    if instrument == "HSP":
        return shf_lookup["OPMODE"].strip()
    # For most HST instrumnents, this should work...
    try:
        return lookup["OBSMODE"].strip()
    except KeyError:
        pass
    raise ValueError("instrument_mode_id not found for " + fname(lookup))


##############################
# get_mast_observation_id
##############################
def get_mast_observation_id(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<mast_observation_id />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        return lookup["ROOTNAME"].strip().lower()
    except KeyError:
        return lookup["ASN_ID"].strip().lower()  # not sure if this gets used


##############################
# get_moving_target_flag
##############################
def get_moving_target_flag(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<instrument_id />`` XML element.
    """
    # Usually in the first FITS header, but in the shm/shf header for GHRS
    for lookup in (data_lookups[0], shf_lookup):
        try:
            value = lookup["MTFLAG"]
            if value == 0:
                return "false"
            if value == 1:
                return "true"
            # should be a string
            value == value.strip()
            if value == "T":
                return "true"
            if value in ("F", ""):
                return "false"
            else:
                raise ValueError(
                    "unrecognized MTFLAG value (%s) for %s"
                    % (value, fname(data_lookups[0]))
                )
        except KeyError:
            pass
    raise ValueError("missing MTFLAG value for %s" % fname(data_lookups[0]))


##############################
# get_observation_type
##############################
def get_observation_type(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<observation_type />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        obstype = lookup["OBSTYPE"].strip()
        if obstype not in ("IMAGING", "SPECTROGRAPHIC"):
            obstype = ""
    except KeyError:
        obstype = ""
    if not obstype:
        instrument = get_instrument_id(data_lookups, shf_lookup)
        if instrument in ("ACS", "NICMOS", "WFC3", "WF/PC", "WFPC2"):
            obstype = "IMAGING"
        elif instrument in ("COS", "FOS", "GHRS"):
            obstype = "SPECTROGRAPHIC"
        elif instrument == "HSP":
            obstype = "TIME-SERIES"
        else:
            raise ValueError("missing OBSTYPE in " + fname(lookup))
    return obstype


##############################
# get_proposed_aperture_name
##############################
def get_proposed_aperture_name(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<proposed_aperture_name />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        return lookup["PROPAPER"].strip()  # only a few instruments distinguish
        # between proposed and actual
    except KeyError:
        return get_aperture_name(data_lookups, shf_lookup)


##############################
# get_repeat_exposure_count
##############################
def get_repeat_exposure_count(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<repeat_exposure_count />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        return str(lookup["NRPTEXP"])
    except KeyError:
        return "1"


##############################
# get_plate_scale
##############################
PLATE_SCALES = {  # plate scales in arcsec/pixel
    ("ACS", "HRC"): 0.026,
    ("ACS", "SBC"): 0.032,
    ("ACS", "WFC1"): 0.05,
    ("ACS", "WFC2"): 0.05,
    ("FOC", "FOC"): 0.014,
    ("NICMOS", "NIC1"): 0.042,
    ("NICMOS", "NIC2"): 0.075,
    ("NICMOS", "NIC3"): 0.2,
    ("WF/PC", "WFC"): 0.1016,
    ("WF/PC", "PC"): 0.0439,
    ("WFPC2", "PC1"): 0.046,
    ("WFPC2", "WF2"): 0.1,
    ("WFPC2", "WF3"): 0.1,
    ("WFPC2", "WF4"): 0.1,
}


def get_plate_scale(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<plate_scale />`` XML element.
    """
    lookup = data_lookups[0]
    # Works for STIS
    try:
        return str(lookup["PLATESC"])
    except KeyError:
        pass
    # Works for any instrument tabulated in the PLATE_SCALEs dictionary above
    instrument = get_instrument_id(data_lookups, shf_lookup)
    detectors = get_detector_ids(data_lookups, shf_lookup)
    scale = SCALE_MAX = 1.0e99
    for detector in detectors:
        key = (instrument, detector)
        if key in PLATE_SCALES:
            scale = min(scale, PLATE_SCALES[key])
    if scale < SCALE_MAX:
        scale *= int(get_binning_mode(data_lookups, shf_lookup))
        formatted = "%.4f" % scale  # up to 4 decimal places
        return formatted.rstrip("0")  # don't include trailing zeros
    return "0.0"


##############################
# get_spectral_resolution
##############################
def get_spectral_resolution(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<spectral_resolution />`` XML element.
    """
    lookup = data_lookups[0]
    # Works for STIS
    try:
        return "%.4f" % (float(lookup["SPECRES"]) * 1.0e-4)
    except KeyError:
        return "0."


##############################
# get_subarray_flag
##############################
def get_subarray_flag(data_lookups: List[Lookup], shf_lookup: Lookup) -> str:
    """
    Return text for the ``<subarray_flag />`` XML element.
    """
    lookup = data_lookups[0]
    try:
        value = lookup["SUBARRAY"]
        if value == 1 or value.startswith("T"):
            return "true"
        elif value == 0 or value.startswith("F"):
            return "false"
        raise ValueError(
            "unrecognized SUBARRAY value (%s) in %s" % (str(value), fname(lookup))
        )
    except KeyError:
        return "false"


##############################
# get_targeted_detector_id
##############################
def get_targeted_detector_id(
    data_lookups: List[Lookup], shf_lookup: Lookup
) -> List[str]:
    """
    Return a list of one or more text values for the
    ``<targeted_detector_id />`` XML element.
    """
    lookup = data_lookups[0]
    instrument = get_instrument_id(data_lookups, shf_lookup)
    aperture = get_aperture_name(data_lookups, shf_lookup)
    if instrument == "WFPC2":
        if aperture in ("WFALL", "WFALL-FIX"):
            return ["PC1", "WF2", "WF3", "WF4"]
        if aperture in ("PC1", "PC1-FIX", "POLQP15P", "FQCH4P15"):
            return ["PC1"]
        if aperture in (
            "WF2",
            "WF2-FIX",
            "FQUVN33",
            "POLQN33",
            "POLQN18",
            "POLQP15W",
            "FQCH4NW2",
            "FQCH4N33",
        ):
            return ["WF2"]
        if aperture in ("WF3", "WF3-FIX", "FQCH4NW3", "F160BN15"):
            return ["WF3"]
        if aperture in ("WF4", "WF4-FIX", "FQCH4NW4"):
            return ["WF4"]
        if aperture == "FQCH4N1":
            return ["PC1", "WF3"]
        raise ValueError(
            "unrecognized WFPC2 aperture (%s) for %s", (aperture, fname(lookup))
        )
    channel = get_channel_id(data_lookups, shf_lookup)
    if instrument == "ACS" and channel == "WFC":
        if aperture.startswith("WFC1"):
            return ["WFC1"]
        if aperture.startswith("WFC2"):
            return ["WFC2"]
        return ["WFC1", "WFC2"]
    if instrument == "WFC" and channel == "UVIS":
        if aperture.startswith("UVIS1"):
            return ["UVIS1"]
        if aperture.startswith("UVIS2"):
            return ["UVIS2"]
        if aperture.startswith("UVIS-QUAD"):
            filter = lookup["FILTER"].strip()
            if filter in (
                "FQ378N",
                "FQ387N",
                "FQ437N",
                "FQ492N",
                "FQ508N",
                "FQ619N",
                "FQ674N",
                "FQ750N",
                "FQ889N",
                "FQ937N",
            ):
                return ["UVIS1"]
            if filter in (
                "FQ232N",
                "FQ243N",
                "FQ422M",
                "FQ436N",
                "FQ575N",
                "FQ634N",
                "FQ672N",
                "FQ727N",
                "FQ906N",
                "FQ924N",
            ):
                return ["UVIS2"]
            raise ValueError(
                "unrecognized quad aperture/filter (%s/%s) in %s"
                % (aperture, filter, fname(lookup))
            )
        return ["UVIS1", "UVIS2"]
    if instrument == "WF/PC":
        # I cannot find documentation for the apertures for WF/PC so this is
        # just an educated guess
        if aperture not in ("ALL", "W1", "W2", "W3", "W4", "P5", "P6", "P7", "P8"):
            raise ValueError(
                "unknown WF/PC aperture (%s) in %s" % (aperture, fname(lookup))
            )
        if aperture == "ALL":
            return get_detector_ids(data_lookups, shf_lookup)  # all detectors
        elif aperture.startswith("W"):
            return [aperture[0] + "F" + aperture[1]]  # WFn
        else:
            return [aperture[0] + "C" + aperture[1]]  # PCn
    return get_detector_ids(data_lookups, shf_lookup)


##### Mark has not changed anything below this line
##############################
# def get_hst_parameters(data_lookups: List[Lookup]) -> NodeBuilder:
#     """Return an ``<hst:HST />`` XML element."""
#     lookup = data_lookups[0]
#     instrument = lookup["INSTRUME"]
#     d = {
#         "mast_observation_id": get_mast_observation_id(lookup),
#         "hst_proposal_id": get_hst_proposal_id(lookup),
#         "hst_pi_name": get_hst_pi_name(lookup),
#         "hst_target_name": get_hst_target_name(lookup),
#         "aperture_name": get_aperture_name(lookup, instrument),
#         "exposure_duration": get_exposure_duration(start_stop_times),
#         "exposure_type": get_exposure_type(lookup, instrument),
#         "filter_name": get_filter_name(lookup, instrument),
#         "fine_guidance_system_lock_type": get_fine_guidance_system_lock_type(lookup),
#         "instrument_mode_id": get_instrument_mode_id(lookup, instrument),
#         "moving_target_flag": "true",
#     }
#     if instrument == "acs":
#         parameters_instrument = parameters_acs(
#             {
#                 "detector_id": get_detector_id(lookup, instrument),
#                 "gain_mode_id": get_gain_mode_id(lookup, instrument),
#                 "observation_type": get_observation_type(lookup, instrument),
#                 "repeat_exposure_count": get_repeat_exposure_count(lookup),
#                 "subarray_flag": get_subarray_flag(lookup, instrument),
#             }
#         )
#     elif instrument == "wfpc2":
#         parameters_instrument = parameters_wfpc2(
#             {
#                 "bandwidth": get_bandwidth(lookup, instrument),
#                 "center_filter_wavelength": get_center_filter_wavelength(
#                     lookup, instrument
#                 ),
#                 "targeted_detector_id": get_targeted_detector_id(
#                     get_aperture_name(lookup, instrument)
#                 ),
#                 "gain_mode_id": get_gain_mode_id(lookup, instrument),
#             }
#         )
#     elif instrument == "wfc3":
#         parameters_instrument = parameters_wfc3(
#             {
#                 "detector_id": get_detector_id(lookup, instrument),
#                 "observation_type": get_observation_type(lookup, instrument),
#                 "repeat_exposure_count": get_repeat_exposure_count(lookup),
#                 "subarray_flag": get_subarray_flag(lookup, instrument),
#             }
#         )
#     else:
#         assert False, f"Bad instrument value: {instrument}"
#     return hst(
#         {
#             "parameters_general": parameters_general(d),
#             "parameters_instrument": parameters_instrument,
#         }
#     )
