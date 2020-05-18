"""
Functionality to build an ``<hst:HST />`` XML element using a SQLite
database.
"""

from typing import Any, Dict, List

from pdart.labels.HstParametersXml import (
    get_targeted_detector_id,
    hst,
    parameters_acs,
    parameters_general,
    parameters_wfc3,
    parameters_wfpc2,
)
from pdart.xml.Templates import NodeBuilder

_CARDS = List[Dict[str, Any]]

_USING_PLACEHOLDER: bool = True


def get_repeat_exposure_count(card_dicts: _CARDS) -> str:
    """
    Return a placeholder integer for the ``<repeat_exposure_count
    />`` XML element, noting the problem.
    """
    try:
        return card_dicts[0]["NRPTEXP"]
    except KeyError:
        return "1"


def get_subarray_flag(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return text for the ``<subarray_flag />`` XML element.
    """
    if _USING_PLACEHOLDER:
        # TODO-PLACEHOLDER
        return "@@@"
    assert instrument != "wfpc2", instrument
    return card_dicts[0]["SUBARRAY"].lower()


##############################
# get_aperture_name
##############################


def get_aperture_name(
    card_dicts: _CARDS, shm_card_dicts: _CARDS, instrument: str
) -> str:
    """
    Return text for the ``<aperture_name />`` XML element.
    """
    if instrument == "wfpc2":
        try:
            res = shm_card_dicts[0]["APERTURE"]
        except KeyError:
            res = shm_card_dicts[0]["APEROBJ"]
    else:
        res = card_dicts[0]["APERTURE"]
    return res


##############################
# get_bandwidth
##############################


def get_bandwidth(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return a float for the ``<bandwidth />`` XML element.
    """
    assert instrument == "wfpc2", instrument
    bandwid = float(card_dicts[0]["BANDWID"])
    return str(bandwid * 1.0e-4)


##############################
# get_center_filter_wavelength
##############################


def get_center_filter_wavelength(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return a float for the ``<center_filter_wavelength />`` XML element.
    """
    assert instrument == "wfpc2", instrument
    centrwv = float(card_dicts[0]["CENTRWV"])
    return str(centrwv * 1.0e-4)


##############################
# get_detector_id
##############################


def get_detector_id(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return text for the ``<detector_id />`` XML element.
    """

    if _USING_PLACEHOLDER:
        # TODO-PLACEHOLDER
        return "@@@"
    detector = card_dicts[0]["DETECTOR"]
    if instrument == "wfpc2":
        if detector == "1":
            res = "PC1"
        else:
            res = "WF" + detector
    else:
        res = detector
    return res.lower()


##############################
# get_exposure_duration
##############################


def get_exposure_duration(card_dicts: _CARDS) -> str:
    """
    Return a float for the ``<exposure_duration />`` XML element.
    """
    return str(card_dicts[0]["EXPTIME"])


##############################
# get_exposure_type
##############################


def get_exposure_type(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return text for the ``<exposure_type />`` XML element.
    """
    if instrument == "acs":
        try:
            res = card_dicts[0]["EXPFLAG"]
        except KeyError:
            res = "UNK"
    else:
        res = card_dicts[0]["EXPFLAG"]
    return res.lower()


##############################
# get_filter_name
##############################


def get_filter_name(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return text for the ``<filter_name />`` XML element.
    """
    if instrument == "wfpc2":
        filtnam1 = card_dicts[0]["FILTNAM1"].strip()
        filtnam2 = card_dicts[0]["FILTNAM2"].strip()
        if filtnam1 == "":
            res = filtnam2
        elif filtnam2 == "":
            res = filtnam1
        else:
            res = f"{filtnam1}+{filtnam2}"
    elif instrument == "acs":
        filter1 = card_dicts[0]["FILTER1"].strip()
        filter2 = card_dicts[0]["FILTER2"].strip()
        if filter1.startswith("clear"):
            if filter2.startswith("clear"):
                res = "clear"
            else:
                res = filter2
        else:
            if filter2.startswith("clear"):
                res = filter1
            else:
                res = f"{filter1}+{filter2}"
    else:
        assert instrument == "wfc3"
        res = card_dicts[0]["FILTER"]
    return res.lower()


##############################
# get_fine_guidance_system_lock_type
##############################


def get_fine_guidance_system_lock_type(card_dicts: _CARDS) -> str:
    """
    Return text for the ``<fine_guidance_system_lock_type />`` XML element.
    """
    try:
        res = card_dicts[0]["FGSLOCK"]
    except KeyError:
        res = "UNK"
    return res.lower()


##############################
# get_gain_mode_id
##############################


def get_gain_mode_id(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return text for the ``<gain_mode_id />`` XML element.
    """
    if _USING_PLACEHOLDER:
        # TODO-PLACEHOLDER
        return "@@@"
    try:
        atodgain = card_dicts[0]["ATODGAIN"]
        if instrument == "acs":
            res = str(atodgain)
        elif instrument == "wfpc2":
            res = "A2D" + str(int(atodgain))
        else:
            assert False
    except KeyError:
        res = "N/A"
    return res.lower()


##############################
# get_hst_pi_name
##############################


def get_hst_pi_name(
    card_dicts: _CARDS, raw_card_dicts: _CARDS, shm_card_dicts: _CARDS
) -> str:
    """
    Return text for the ``<hst_pi_name />`` XML element.
    """

    def get_from_dicts(dicts: _CARDS) -> str:
        pr_inv_l = dicts[0]["PR_INV_L"]
        pr_inv_f = dicts[0]["PR_INV_F"]
        try:
            pr_inv_m = dicts[0]["PR_INV_M"]
            return f"{pr_inv_l}, {pr_inv_f} {pr_inv_m}"
        except KeyError:
            return f"{pr_inv_l}, {pr_inv_f}"

    try:
        return get_from_dicts(card_dicts)
    except:
        try:
            return get_from_dicts(raw_card_dicts)
        except:
            return get_from_dicts(shm_card_dicts)


##############################
# get_hst_proposal_id
##############################


def get_hst_proposal_id(card_dicts: _CARDS) -> str:
    """
    Return text for the ``<hst_proposal_id />`` XML element.
    """
    return str(card_dicts[0]["PROPOSID"])


##############################
# get_hst_target_name
##############################


def get_hst_target_name(card_dicts: _CARDS) -> str:
    """
    Return text for the ``<hst_target_name />`` XML element.
    """
    return card_dicts[0]["TARGNAME"]


##############################
# get_instrument_mode_id
##############################


def get_instrument_mode_id(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return text for the ``<instrument_mode_id />`` XML element.
    """
    if instrument == "acs":
        try:
            res = card_dicts[0]["OBSMODE"]
        except KeyError:
            res = "UNK"
    elif instrument == "wfpc2":
        res = card_dicts[0]["MODE"]
    else:
        assert instrument == "wfc3"
        res = card_dicts[0]["OBSMODE"]
    res = res.lower()

    # TODO Temporary hack until I get updated XML schemas to allow 'multiaccum' as a value
    if _USING_PLACEHOLDER and res == "multiaccum":
        # TODO-PLACEHOLDER
        res = "accum"

    return res


##############################
# get_observation_type
##############################


def get_observation_type(card_dicts: _CARDS, instrument: str) -> str:
    """
    Return text for the ``<observation_type />`` XML element.
    """
    assert instrument != "wfpc2"
    raw_value = card_dicts[0]["OBSTYPE"].lower()
    if raw_value == "imaging":
        return "image"
    else:
        return raw_value


##############################
# get_mast_observation_id:
##############################


def get_mast_observation_id(card_dicts: _CARDS) -> str:
    """
    Return text for the ``<mast_observation_id />`` XML element.
    """
    try:
        asn_id = card_dicts[0]["ASN_ID"]
        if asn_id == "NONE":
            res = card_dicts[0]["ROOTNAME"]
        else:
            res = asn_id
    except KeyError:
        res = card_dicts[0]["ROOTNAME"]
    return res


##############################


def get_hst_parameters(
    card_dicts: _CARDS, raw_card_dicts: _CARDS, shm_card_dicts: _CARDS, instrument: str
) -> NodeBuilder:
    """Return an ``<hst:HST />`` XML element."""
    d = {
        "mast_observation_id": get_mast_observation_id(card_dicts),
        "hst_proposal_id": get_hst_proposal_id(card_dicts),
        "hst_pi_name": get_hst_pi_name(card_dicts, raw_card_dicts, shm_card_dicts),
        "hst_target_name": get_hst_target_name(card_dicts),
        "aperture_name": get_aperture_name(card_dicts, shm_card_dicts, instrument),
        "exposure_duration": get_exposure_duration(card_dicts),
        "exposure_type": get_exposure_type(card_dicts, instrument),
        "filter_name": get_filter_name(card_dicts, instrument),
        "fine_guidance_system_lock_type": get_fine_guidance_system_lock_type(
            card_dicts
        ),
        "instrument_mode_id": get_instrument_mode_id(card_dicts, instrument),
        "moving_target_flag": "true",
    }

    if instrument == "acs":
        parameters_instrument = parameters_acs(
            {
                "detector_id": get_detector_id(card_dicts, instrument),
                "gain_mode_id": get_gain_mode_id(card_dicts, instrument),
                "observation_type": get_observation_type(card_dicts, instrument),
                "repeat_exposure_count": get_repeat_exposure_count(card_dicts),
                "subarray_flag": get_subarray_flag(card_dicts, instrument),
            }
        )
    elif instrument == "wfpc2":
        parameters_instrument = parameters_wfpc2(
            {
                "bandwidth": get_bandwidth(card_dicts, instrument),
                "center_filter_wavelength": get_center_filter_wavelength(
                    card_dicts, instrument
                ),
                "targeted_detector_id": get_targeted_detector_id(
                    get_aperture_name(card_dicts, shm_card_dicts, instrument)
                ),
                "gain_mode_id": get_gain_mode_id(card_dicts, instrument),
            }
        )
    elif instrument == "wfc3":
        parameters_instrument = parameters_wfc3(
            {
                "detector_id": get_detector_id(card_dicts, instrument),
                "observation_type": get_observation_type(card_dicts, instrument),
                "repeat_exposure_count": get_repeat_exposure_count(card_dicts),
                "subarray_flag": get_subarray_flag(card_dicts, instrument),
            }
        )
    else:
        assert False, f"Bad instrument value: {instrument}"

    return hst(
        {
            "parameters_general": parameters_general(d),
            "parameters_instrument": parameters_instrument,
        }
    )
