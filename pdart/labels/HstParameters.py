"""
Functionality to build an ``<hst:HST />`` XML element using a SQLite
database.
"""
import julian

from typing import Any, Dict, List, Tuple

from pdart.labels.HstParametersXml import (
    detector_id,
    moving_target_description,
    moving_target_keyword,
    targeted_detector_id,
    hst_parameters,
    program_parameters,
    instrument_parameters,
    pointing_parameters,
    tracking_parameters,
    exposure_parameters,
    wavelength_filter_grating_parameters,
    operational_parameters,
)
from pdart.labels.Lookup import Lookup, merge_two_hdu_lookups
from pdart.xml.Templates import (
    FragBuilder,
    NodeBuilder,
    NodeBuilderTemplate,
    combine_nodes_into_fragment,
)

# All functions have the same input arguments:
#   data_lookups: List[Lookup]
#           a list of all the FITS headers in a data file (raw, d0f, drz, etc.)
#   shm_lookup: Lookup
#           the first fits header of the associated _shm.fits file
#           or _spt.fits file.
# The second argument is needed because sometimes the data file does not contain
# all the info we need.


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
def get_aperture_name(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<aperture_name />`` XML element.
    """
    instrument = get_instrument_id(data_lookups, shm_lookup)
    if instrument in ("WF/PC", "WFPC2", "HSP"):
        return shm_lookup["APER_1"].strip()
    if instrument == "FOS":
        return shm_lookup["APER_ID"]
    # This is valid for most instruments
    try:
        return data_lookups[0]["APERTURE"].strip()
    except KeyError:
        pass
    raise ValueError("missing aperture for " + fname(shm_lookup))


##############################
# get_bandwidth
##############################
def get_bandwidth(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return a float for the ``<bandwidth />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    # Works for STIS and WFPC2
    try:
        return "%.4f" % (float(lookup["BANDWID"]) * 1.0e-4)
    except KeyError:
        return "0."


##############################
# get_binning_mode
##############################
def get_binning_mode(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<binning_mode />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    instrument = get_instrument_id(data_lookups, shm_lookup)
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
def get_center_filter_wavelength(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return a float for the ``<center_filter_wavelength />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    # Works for STIS and WFPC2
    try:
        return "%.4f" % (float(lookup["CENTRWV"]) * 1.0e-4)
    except KeyError:
        return "0."


##############################
# get_channel_id
##############################
def get_channel_id(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<channel_id />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    instrument = get_instrument_id(data_lookups, shm_lookup)
    if instrument == "NICMOS":
        result = "NIC" + str(lookup["CAMERA"])
    elif instrument == "WF/PC":
        result = lookup["CAMERA"].strip()
    else:
        try:
            return lookup["DETECTOR"].strip()
        except KeyError:
            result = instrument

    return result


##############################
# get_coronagraph_flag
##############################
def get_coronagraph_flag(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<coronagraph_flag />`` XML element.
    """
    instrument = get_instrument_id(data_lookups, shm_lookup)
    aperture = get_aperture_name(data_lookups, shm_lookup)
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
def get_cosmic_ray_split_count(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<cosmic_ray_split_count />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    try:
        return str(lookup["CRSPLIT"])
    except KeyError:
        return "1"  # no CR-splitting unless explicitly stated


##############################
# get_detector_ids
##############################
WFPC2_DETECTOR_IDS = {1: "PC1", 2: "WF2", 3: "WF3", 4: "WF4"}


def get_detector_ids(data_lookups: List[Lookup], shm_lookup: Lookup) -> List[str]:
    """
    Return a list of zero or more text values for the ``<detector_id />``
    XML elements.
    """
    # Interior function
    def get_ccds_from_lookups(data_lookups: List[Lookup], fitsname: str) -> List[int]:
        ccds: List[int] = []
        for lookup in data_lookups:
            try:
                ccdchip = int(lookup[fitsname])
                ccds.append(ccdchip)
            except KeyError:
                pass
        ccds = list(set(ccds))  # select unique values
        ccds.sort()
        return ccds

    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    instrument = get_instrument_id(data_lookups, shm_lookup)
    channel = get_channel_id(data_lookups, shm_lookup)
    if instrument == "ACS" and channel == "WFC":
        ccds = get_ccds_from_lookups(data_lookups, "CCDCHIP")
        if -999 in ccds:
            ccds = [1, 2]
        result = [f"WFC{k}" for k in ccds]
    elif instrument == "COS" and channel == "FUV":
        segment = lookup["SEGMENT"].strip()
        if segment not in ("FUVA", "FUVB", "BOTH"):
            raise ValueError(
                "unrecognized segment (%s) in %s" % (segment, fname(lookup))
            )
        if segment == "FUVA":
            result = ["FUVA"]
        elif segment == "FUVB":
            result = ["FUVB"]
        else:
            result = ["FUVA", "FUVB"]
    elif instrument == "GHRS":
        result = ["GHRS" + str(lookup["DETECTOR"])]
    elif instrument == "HSP":
        config = shm_lookup["CONFIG"].strip()
        # Example: config = HSP/UNK/VIS
        parts = config.split("/")
        if parts[0] != "HSP":
            raise ValueError(f"Invalid CONFIG value in {fname(lookup)}.")
        result = [p for p in parts[1:] if p != "UNK"]
    elif instrument == "WFC3" and channel == "UVIS":
        ccds = get_ccds_from_lookups(data_lookups, "CCDCHIP")
        if -999 in ccds:
            ccds = [1, 2]
        result = [f"UVIS{k}" for k in ccds]
    elif instrument == "WF/PC":
        # We will need to find a workaround to read the FITS table from the data
        # file, because that is the only way to get the actual set of detectors
        # if there are less than four! I hope it just doesn't come up.
        count = lookup["NAXIS3"]
        if count != 4:
            raise ValueError(
                "unknown detector subset in (%d/4) in %s" % (count, fname(lookup))
            )
        if channel not in ("PC", "WFC"):
            raise ValueError(f"Bad channel for {fname(lookup)}.")
        if channel == "WFC":
            result = ["WF1", "WF2", "WF3", "WF4"]
        else:
            result = ["PC5", "PC6", "PC7", "PC8"]
    elif instrument == "WFPC2":
        ccds = get_ccds_from_lookups(data_lookups, "DETECTOR")
        result = [WFPC2_DETECTOR_IDS[k] for k in ccds]
    # Otherwise, return the single value of channel_id
    else:
        result = [channel]

    return result


##############################
# get_exposure_duration
##############################
def get_exposure_duration(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return a float for the ``<exposure_duration />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    try:
        return str(lookup["EXPTIME"])
    except KeyError:
        return str(lookup["TEXPTIME"])


##############################
# get_exposure_type
##############################
def get_exposure_type(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<exposure_type />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    return lookup["EXPFLAG"].strip()


##############################
# get_filter_name
##############################
def get_filter_name(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<filter_name />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    instrument = get_instrument_id(data_lookups, shm_lookup)
    if instrument == "ACS":
        filter1 = lookup["FILTER1"].strip()
        filter2 = lookup["FILTER2"].strip()
        if filter1.startswith("CLEAR"):
            if filter2.startswith("CLEAR") or filter2 == "N/A":
                return "CLEAR"
            else:
                return filter2
        if filter2.startswith("CLEAR") or filter2 == "N/A":
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
        return shm_lookup["SPEC_1"].strip()
    if instrument == "GHRS":
        return lookup["GRATING"].strip()
    if instrument == "STIS":
        opt_elem = lookup["OPT_ELEM"].strip()
        filter = lookup["FILTER"].strip().upper().replace(" ", "_")
        if filter == "CLEAR":
            return opt_elem
        else:
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
    data_lookups: List[Lookup], shm_lookup: Lookup
) -> str:
    """
    Return text for the ``<fine_guidance_system_lock_type />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    return lookup["FGSLOCK"].strip()


##############################
# get_gain_setting
##############################
def get_gain_setting(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<gain_mode_id />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    # Works for WFPC2
    try:
        wfpc2_gain: int = int(float(lookup["ATODGAIN"]))  # format WFPC2 gains as ints
        if wfpc2_gain in (7, 15):
            return str(wfpc2_gain)
        raise ValueError(
            "unrecognized WFPC2 gain (%d) in %s" % (wfpc2_gain, fname(lookup))
        )
    except KeyError:
        pass
    # Works for ACS, WFC3, others
    try:
        gain: float = float(lookup["CCDGAIN"])
        return "%3.1f" % gain  # format other gains with one decimal
    except KeyError:
        pass
    return "0."


##############################
# get_gyroscope_mode
##############################
def get_gyroscope_mode(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<gyroscope_mode />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    try:
        return str(lookup["GYROMODE"]).strip().replace("T", "3")
    except KeyError:
        return "3"  # Three-gyro mode unless otherwise specified


##############################
# get_hst_pi_name
##############################
def get_hst_pi_name(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<hst_pi_name />`` XML element.
    """
    # Usually in the first FITS header, but in the shm header for GHRS
    for lookup in (data_lookups[0], shm_lookup):
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
def get_hst_proposal_id(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<hst_proposal_id />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    return str(lookup["PROPOSID"])


##############################
# get_hst_target_name
##############################
def get_hst_target_name(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<hst_target_name />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    return lookup["TARGNAME"]


##############################
# get_instrument_id
##############################
def get_instrument_id(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<instrument_id />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    instrument = lookup["INSTRUME"].strip()
    if instrument == "HRS":
        return "GHRS"
    if instrument == "WFPC":
        return "WF/PC"
    return instrument


##############################
# get_instrument_mode_id
##############################
def get_instrument_mode_id(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<instrument_mode_id />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    instrument = get_instrument_id(data_lookups, shm_lookup)
    if instrument in ("WF/PC", "WFPC2"):
        return lookup["MODE"].strip()
    if instrument == "FOC":
        return lookup["OPTCRLY"].strip()
    if instrument == "HSP":
        return shm_lookup["OPMODE"].strip()
    # For most HST instrumnents, this should work...
    try:
        return lookup["OBSMODE"].strip()
    except KeyError:
        pass
    raise ValueError("instrument_mode_id not found for " + fname(lookup))


##############################
# get_mast_observation_id
##############################
def get_mast_observation_id(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<mast_observation_id />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    KEYS = ["ROOTNAME", "ASN_ID"]
    for keyword in KEYS:
        try:
            return lookup[keyword].strip().lower()
        except KeyError:
            pass

    raise RuntimeError(f"lookup = {lookup}, shm_lookup = {shm_lookup}")


##############################
# get_moving_target_descriptions
##############################
def get_moving_target_descriptions(
    data_lookups: List[Lookup], shm_lookup: Lookup
) -> List[str]:
    """
    Return text for the ``<moving_target_description />`` XML element.
    """
    # Defined by "MT_LV_n" keyword values in the _shm.fits files
    descs = []
    for k in range(1, 10):
        keyword = "MT_LV_" + str(k)
        try:
            value = shm_lookup[keyword].strip()
            descs.append(value)
        except KeyError:
            break
    if descs:
        return descs
    else:
        return ["Not applicable"]


##############################
# get_moving_target_keywords
##############################
def get_moving_target_keywords(
    data_lookups: List[Lookup], shm_lookup: Lookup
) -> List[str]:
    """
    Return text for the ``<moving_target_keywords />`` XML element.
    """
    # Defined by "TARKEYn" keyword values in the _shm.fits files
    keywords = []
    for k in range(1, 10):
        keyword = "TARKEY" + str(k)
        try:
            value = shm_lookup[keyword].strip()
            keywords.append(value)
        except KeyError:
            break
    if keywords:
        return keywords
    else:
        return ["Not applicable"]


##############################
# get_moving_target_flag
##############################
def get_moving_target_flag(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<instrument_id />`` XML element.
    """
    # Usually in the first FITS header, but in the shm header for GHRS
    for lookup in (data_lookups[0], shm_lookup):
        try:
            value = lookup["MTFLAG"].strip()
            if value in ("T", "1"):
                return "true"
            if value in ("F", "", "0"):
                return "false"
            else:
                raise ValueError(
                    f"unrecognized MTFLAG value ({value}: {type(value)}) for {fname(data_lookups[0])}"
                )
        except KeyError:
            pass
    raise ValueError("missing MTFLAG value for %s" % fname(data_lookups[0]))


##############################
# get_observation_type
##############################
def get_observation_type(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<observation_type />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    try:
        obstype = lookup["OBSTYPE"].strip()
        if obstype not in ("IMAGING", "SPECTROGRAPHIC"):
            obstype = ""
    except KeyError:
        obstype = ""
    if not obstype:
        instrument = get_instrument_id(data_lookups, shm_lookup)
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
def get_proposed_aperture_name(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<proposed_aperture_name />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    try:
        res = lookup["PROPAPER"].strip()  # only a few instruments distinguish
        if res:
            return res
        else:
            return get_aperture_name(data_lookups, shm_lookup)
    except KeyError:
        return get_aperture_name(data_lookups, shm_lookup)


##############################
# get_repeat_exposure_count
##############################
def get_repeat_exposure_count(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<repeat_exposure_count />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
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


def get_plate_scale(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<plate_scale />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    # Works for STIS
    try:
        return str(lookup["PLATESC"])
    except KeyError:
        pass
    # Works for any instrument tabulated in the PLATE_SCALEs dictionary above
    instrument = get_instrument_id(data_lookups, shm_lookup)
    detectors = get_detector_ids(data_lookups, shm_lookup)
    scale = SCALE_MAX = 1.0e99
    for detector in detectors:
        key = (instrument, detector)
        if key in PLATE_SCALES:
            scale = min(scale, PLATE_SCALES[key])
    if scale < SCALE_MAX:
        scale *= int(get_binning_mode(data_lookups, shm_lookup))
        formatted = "%.4f" % scale  # up to 4 decimal places
        return formatted.rstrip("0")  # don't include trailing zeros
    return "0.0"


##############################
# get_spectral_resolution
##############################
def get_spectral_resolution(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<spectral_resolution />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    # Works for STIS
    try:
        return "%.4f" % (float(lookup["SPECRES"]) * 1.0e-4)
    except KeyError:
        return "0."


##############################
# get_start_stop_date_time
##############################
def get_start_stop_date_times(
    data_lookups: List[Lookup], shm_lookup: Lookup
) -> Tuple[str, str]:
    """
    Return text for the ``<start_date_time />`` and ``<stop_date_time />`` XML
    elements.
    """

    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])

    # HST documents indicate that times are only accurate to a second or so.
    # This is consistent with the fact that start times indicated by DATE-OBS
    # and TIME-OBS often disagree with the times as indicated by EXPSTART at the
    # level of a second or so. For any individual time, this is fine, but we
    # want to be sure that the difference between the start and stop times is
    # compatible with the exposure time, whenever appropriate.
    #
    # I say "whenever appropriate" because there are times when multiple images
    # have been drizzled or otherwise merged. In this case, the start and stop
    # times refer to the first and last of the set of images, respectively, and
    # their difference can be much greater than the exposure time.
    #
    # It takes some careful handling to get the behavior we want.

    # Figure out what's available in the header
    try:
        date_obs = lookup["DATE-OBS"]
    except KeyError:
        date_obs = None

    try:
        time_obs = lookup["TIME-OBS"]
    except KeyError:
        time_obs = None

    exptime = float(lookup["EXPTIME"])

    try:  # either EXPSTART or TEXPSTART should be available
        expstart = float(lookup["EXPSTART"])
    except KeyError:
        expstart = float(lookup["TEXPSTART"])

    try:  # either EXPEND or TEXPEND should be available
        expend = float(lookup["EXPEND"])
    except KeyError:
        expend = float(lookup["TEXPEND"])

    # Decide which delta-time to use
    # Our start and stop times are only ever good to the nearest second, but we
    # want to ensure that the difference looks right. For this purpose,
    # non-integral exposure times should be rounded up to the next integer.
    delta_from_mjd = (expend - expstart) * 86400.0
    if delta_from_mjd > exptime + 2.0:  # if the delta is too large, we know
        # multiple images were combined
        delta = delta_from_mjd
    else:
        delta = -(-exptime // 1.0)  # rounded up to nearest int

    # Fill in the start time; update the expstart in MJD units if necessary.
    # If DATE-OBS and TIME-OBS values are provided, we use this as the start
    # time because it is the value our users would expect. There exist cases
    # when these values are not provided, and in that case we use EXPSTART,
    # converted from MJD. Note that these MJD values are in UTC, not TAI. In
    # other words, we need to ignore leapseconds in these time conversions.
    if date_obs and time_obs:
        start_time = date_obs + "T" + time_obs + "Z"
        day = julian.day_from_iso(date_obs)
        sec = julian.sec_from_iso(time_obs)
        expstart = julian.mjd_from_day_sec(day, sec)
    else:
        (day, sec) = julian.day_sec_from_mjd(expstart)
        start_time = julian.ymdhms_format_from_day_sec(day, sec, suffix="Z")

    # Fill in the stop time. We ensure that this differs from the start time by
    # the expected amount.
    expend = expstart + delta / 86400.0
    (day, sec) = julian.day_sec_from_mjd(expend)
    stop_time = julian.ymdhms_format_from_day_sec(day, sec, suffix="Z")

    return (start_time, stop_time)


##############################
# get_subarray_flag
##############################
def get_subarray_flag(data_lookups: List[Lookup], shm_lookup: Lookup) -> str:
    """
    Return text for the ``<subarray_flag />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    try:
        value: str = lookup["SUBARRAY"]
        if value == "1" or value.startswith("T"):
            return "true"
        elif value == "0" or value.startswith("F"):
            return "false"
        raise ValueError(
            "unrecognized SUBARRAY value (%s) in %s" % (value, fname(lookup))
        )
    except KeyError:
        return "false"


##############################
# get_targeted_detector_ids
##############################
def get_targeted_detector_ids(
    data_lookups: List[Lookup], shm_lookup: Lookup
) -> List[str]:
    """
    Return a list of one or more text values for the
    ``<targeted_detector_ids />`` XML element.
    """
    lookup = merge_two_hdu_lookups(data_lookups[0], data_lookups[1])
    instrument = get_instrument_id(data_lookups, shm_lookup)
    aperture = get_aperture_name(data_lookups, shm_lookup)
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
        ):
            return ["WF2"]
        if aperture == "FQCH4N33":
            return ["WF2", "WF3"]
        if aperture in ("WF3", "WF3-FIX", "FQCH4NW3", "F160BN15"):
            return ["WF3"]
        if aperture in ("WF4", "WF4-FIX", "FQCH4NW4"):
            return ["WF4"]
        if aperture == "FQCH4N1":
            return ["PC1", "WF3"]
        if aperture == "FQCH4N15":
            return ["PC1"]
        if aperture == "FQCH4W3":
            return ["WF3"]
        raise ValueError(
            "unrecognized WFPC2 aperture (%s) for %s [%s]",
            (aperture, fname(lookup), lookup),
        )
    channel = get_channel_id(data_lookups, shm_lookup)
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
            return get_detector_ids(data_lookups, shm_lookup)  # all detectors
        elif aperture.startswith("W"):
            return [aperture[0] + "F" + aperture[1]]  # WFn
        else:
            return [aperture[0] + "C" + aperture[1]]  # PCn
    return get_detector_ids(data_lookups, shm_lookup)


############################################################


def _make_fragment(
    param_name: str, param_values: List[str], node_builder: NodeBuilderTemplate
) -> FragBuilder:
    return combine_nodes_into_fragment(
        [node_builder({param_name: value}) for value in param_values]
    )


def _get_detector_ids_fragment(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> FragBuilder:
    return _make_fragment(
        "detector_id", get_detector_ids(data_lookup, shm_lookup), detector_id
    )


def _get_moving_target_descriptions_fragment(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> FragBuilder:
    return _make_fragment(
        "moving_target_description",
        get_moving_target_descriptions(data_lookup, shm_lookup),
        moving_target_description,
    )


def _get_moving_target_keywords_fragment(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> FragBuilder:
    return _make_fragment(
        "moving_target_keyword",
        get_moving_target_keywords(data_lookup, shm_lookup),
        moving_target_keyword,
    )


def _get_targeted_detector_ids_fragment(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> FragBuilder:
    return _make_fragment(
        "targeted_detector_id",
        get_targeted_detector_ids(data_lookup, shm_lookup),
        targeted_detector_id,
    )


############################################################


def _get_program_parameters(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    return {
        "mast_observation_id": get_mast_observation_id(data_lookup, shm_lookup),
        "hst_proposal_id": get_hst_proposal_id(data_lookup, shm_lookup),
        "hst_pi_name": get_hst_pi_name(data_lookup, shm_lookup),
    }


def _get_instrument_parameters(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    return {
        "instrument_id": get_instrument_id(data_lookup, shm_lookup),
        "channel_id": get_channel_id(data_lookup, shm_lookup),
        "detector_ids": _get_detector_ids_fragment(data_lookup, shm_lookup),  # FRAGMENT
        "observation_type": get_observation_type(data_lookup, shm_lookup),
    }


def _get_pointing_parameters(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    return {
        "hst_target_name": get_hst_target_name(data_lookup, shm_lookup),
        "moving_target_flag": get_moving_target_flag(data_lookup, shm_lookup),
        "moving_target_keywords": _get_moving_target_keywords_fragment(
            data_lookup, shm_lookup
        ),  # FRAGMENT
        "moving_target_descriptions": _get_moving_target_descriptions_fragment(
            data_lookup, shm_lookup
        ),  # FRAGMENT
        "aperture_name": get_aperture_name(data_lookup, shm_lookup),
        "proposed_aperture_name": get_proposed_aperture_name(data_lookup, shm_lookup),
        "targeted_detector_ids": _get_targeted_detector_ids_fragment(
            data_lookup, shm_lookup
        ),  # FRAGMENT
    }


def _get_tracking_parameters(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    return {
        "fine_guidance_sensor_lock_type": get_fine_guidance_sensor_lock_type(
            data_lookup, shm_lookup
        ),
        "gyroscope_mode": get_gyroscope_mode(data_lookup, shm_lookup),
    }


def _get_exposure_parameters(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    return {
        "exposure_duration": get_exposure_duration(data_lookup, shm_lookup),
        "exposure_type": get_exposure_type(data_lookup, shm_lookup),
    }


def _get_wavelength_filter_grating_parameters(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    return {
        "filter_name": get_filter_name(data_lookup, shm_lookup),
        "center_filter_wavelength": get_center_filter_wavelength(
            data_lookup, shm_lookup
        ),
        "bandwidth": get_bandwidth(data_lookup, shm_lookup),
        "spectral_resolution": get_spectral_resolution(data_lookup, shm_lookup),
    }


def _get_operational_parameters(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    return {
        "instrument_mode_id": get_instrument_mode_id(data_lookup, shm_lookup),
        "gain_setting": get_gain_setting(data_lookup, shm_lookup),
        "coronagraph_flag": get_coronagraph_flag(data_lookup, shm_lookup),
        "cosmic_ray_split_count": get_cosmic_ray_split_count(data_lookup, shm_lookup),
        "repeat_exposure_count": get_repeat_exposure_count(data_lookup, shm_lookup),
        "subarray_flag": get_subarray_flag(data_lookup, shm_lookup),
        "binning_mode": get_binning_mode(data_lookup, shm_lookup),
        "plate_scale": get_plate_scale(data_lookup, shm_lookup),
    }


def get_hst_parameters_dict(
    data_lookup: List[Lookup], shm_lookup: Lookup
) -> Dict[Any, Any]:
    sub_dicts: List[Dict[Any, Any]] = [
        _get_program_parameters(data_lookup, shm_lookup),
        _get_instrument_parameters(data_lookup, shm_lookup),
        _get_pointing_parameters(data_lookup, shm_lookup),
        _get_tracking_parameters(data_lookup, shm_lookup),
        _get_exposure_parameters(data_lookup, shm_lookup),
        _get_wavelength_filter_grating_parameters(data_lookup, shm_lookup),
        _get_operational_parameters(data_lookup, shm_lookup),
    ]
    return {key: val for d in sub_dicts for key, val in d.items()}


############################################################
def get_hst_parameters(data_lookup: List[Lookup], shm_lookup: Lookup) -> NodeBuilder:
    d: Dict[Any, Any] = get_hst_parameters_dict(data_lookup, shm_lookup)
    return hst_parameters(
        {
            "program_parameters": program_parameters(d),
            "instrument_parameters": instrument_parameters(d),
            "pointing_parameters": pointing_parameters(d),
            "tracking_parameters": tracking_parameters(d),
            "exposure_parameters": exposure_parameters(d),
            "wavelength_filter_grating_parameters": wavelength_filter_grating_parameters(
                d
            ),
            "operational_parameters": operational_parameters(d),
        }
    )
