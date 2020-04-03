"""
Functionality to build an ``<hst:HST />`` XML element using a SQLite
database.
"""

from pdart.labels.HstParametersXml import (
    get_targeted_detector_id,
    hst,
    parameters_acs,
    parameters_general,
    parameters_wfc3,
    parameters_wfpc2,
)

from typing import Any, Callable, Dict, List, Optional
from pdart.xml.Templates import NodeBuilder

_CARDS = List[Dict[str, Any]]


def get_repeat_exposure_count(product_id: str) -> str:
    """
    Return a placeholder integer for the ``<repeat_exposure_count
    />`` XML element, noting the problem.
    """
    return "0"
    # TODO Ask Mark and implement this.


def get_subarray_flag(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return text for the ``<subarray_flag />`` XML element.
    """
    if instrument != "wfpc2":
        return card_dicts[0]["SUBARRAY"]
    assert False


##############################
# get_aperture_name
##############################


def get_aperture_name(card_dicts, shm_card_dicts, instrument, product_id):
    # type: (_CARDS, _CARDS, str, str) -> str
    """
    Return text for the ``<aperture_name />`` XML element.
    """
    if instrument == "wfpc2":
        try:
            return shm_card_dicts[0]["APERTURE"]
        except KeyError:
            return shm_card_dicts[0]["APEROBJ"]
    else:
        return card_dicts[0]["APERTURE"]


##############################
# get_bandwidth
##############################


def get_bandwidth(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return a float for the ``<bandwidth />`` XML element.
    """
    if instrument == "wfpc2":
        bandwid = float(card_dicts[0]["BANDWID"])
        return str(bandwid * 1.0e-4)
    assert False


##############################
# get_center_filter_wavelength
##############################


def get_center_filter_wavelength(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return a float for the ``<center_filter_wavelength />`` XML element.
    """
    if instrument == "wfpc2":
        centrwv = float(card_dicts[0]["CENTRWV"])
        return str(centrwv * 1.0e-4)
    else:
        raise Exception("Unhandled instrument %s" % instrument)


##############################
# get_detector_id
##############################


def get_detector_id(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return text for the ``<detector_id />`` XML element.
    """

    detector = card_dicts[0]["DETECTOR"]
    if instrument == "wfpc2":
        if detector == "1":
            return "PC1"
        else:
            return "WF" + detector
    else:
        return detector


##############################
# get_exposure_duration
##############################


def get_exposure_duration(card_dicts, product_id):
    # type: (_CARDS, str) -> str
    """
    Return a float for the ``<exposure_duration />`` XML element.
    """
    return str(card_dicts[0]["EXPTIME"])


##############################
# get_exposure_type
##############################


def get_exposure_type(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return text for the ``<exposure_type />`` XML element.
    """
    if instrument == "acs":
        try:
            return card_dicts[0]["EXPFLAG"]
        except KeyError:
            return "UNK"
    else:
        return card_dicts[0]["EXPFLAG"]


##############################
# get_filter_name
##############################


def get_filter_name(card_dicts, instrument, product_name):
    # type: (_CARDS, str, str) -> str
    """
    Return text for the ``<filter_name />`` XML element.
    """
    if instrument == "wfpc2":
        filtnam1 = card_dicts[0]["FILTNAM1"].strip()
        filtnam2 = card_dicts[0]["FILTNAM2"].strip()
        if filtnam1 == "":
            return filtnam2
        elif filtnam2 == "":
            return filtnam1
        else:
            return "%s+%s" % (filtnam1, filtnam2)
    elif instrument == "acs":
        filter1 = card_dicts[0]["FILTER1"].strip()
        filter2 = card_dicts[0]["FILTER2"].strip()
        if filter1.startswith("CLEAR"):
            if filter2.startswith("CLEAR"):
                return "CLEAR"
            else:
                return filter2
        else:
            if filter2.startswith("CLEAR"):
                return filter1
            else:
                return "%s+%s" % (filter1, filter2)
    else:
        assert instrument == "wfc3"
        return card_dicts[0]["FILTER"]


##############################
# get_fine_guidance_system_lock_type
##############################


def get_fine_guidance_system_lock_type(card_dicts, product_id):
    """
    Return text for the ``<fine_guidance_system_lock_type />`` XML element.
    """
    try:
        return card_dicts[0]["FGSLOCK"]
    except KeyError:
        return "UNK"


##############################
# get_gain_mode_id
##############################


def get_gain_mode_id(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return text for the ``<gain_mode_id />`` XML element.
    """
    try:
        atodgain = card_dicts[0]["ATODGAIN"]
    except KeyError:
        return "N/A"
    if instrument == "acs":
        return str(atodgain)
    elif instrument == "wfpc2":
        return "A2D" + str(int(atodgain))
    assert False


##############################
# get_hst_pi_name
##############################


def get_hst_pi_name(card_dicts, product_id):
    # type: (_CARDS, str) -> str
    """
    Return text for the ``<hst_pi_name />`` XML element.
    """
    pr_inv_l = card_dicts[0]["PR_INV_L"]
    pr_inv_f = card_dicts[0]["PR_INV_F"]
    try:
        pr_inv_m = card_dicts[0]["PR_INV_M"]
        return "%s, %s %s" % (pr_inv_l, pr_inv_f, pr_inv_m)
    except KeyError:
        return "%s, %s" % (pr_inv_l, pr_inv_f)


##############################
# get_hst_proposal_id
##############################


def get_hst_proposal_id(card_dicts, product_id):
    # type: (_CARDS, str) -> str
    """
    Return text for the ``<hst_proposal_id />`` XML element.
    """
    return str(card_dicts[0]["PROPOSID"])


##############################
# get_hst_target_name
##############################


def get_hst_target_name(card_dicts, product_id):
    # type: (_CARDS, str) -> str
    """
    Return text for the ``<hst_target_name />`` XML element.
    """
    return card_dicts[0]["TARGNAME"]


##############################
# get_instrument_mode_id
##############################


def get_instrument_mode_id(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return text for the ``<instrument_mode_id />`` XML element.
    """
    if instrument == "acs":
        try:
            return card_dicts[0]["OBSMODE"]
        except KeyError:
            return "UNK"
    if instrument == "wfpc2":
        return card_dicts[0]["MODE"]
    else:
        assert instrument == "wfc3"
        return card_dicts[0]["OBSMODE"]


##############################
# get_observation_type
##############################


def get_observation_type(card_dicts, instrument, product_id):
    # type: (_CARDS, str, str) -> str
    """
    Return text for the ``<observation_type />`` XML element.
    """
    if instrument != "wfpc2":
        return card_dicts[0]["OBSTYPE"]
    else:
        raise Exception("Unhandled instrument %s" % instrument)


##############################
# get_mast_observation_id:
##############################


def get_mast_observation_id(card_dicts, product_id):
    """
    Return text for the ``<mast_observation_id />`` XML element.
    """
    try:
        asn_id = card_dicts[0]["ASN_ID"]
        if asn_id == "NONE":
            return card_dicts[0]["ROOTNAME"]
        else:
            return asn_id
    except KeyError:
        return card_dicts[0]["ROOTNAME"]


##############################


def get_hst_parameters(card_dicts, shm_card_dicts, instrument, product_id):
    # type: (_CARDS, _CARDS, str, str) -> NodeBuilder
    """Return an ``<hst:HST />`` XML element."""
    d = {
        "mast_observation_id": get_mast_observation_id(card_dicts, product_id),
        "hst_proposal_id": get_hst_proposal_id(card_dicts, product_id),
        "hst_pi_name": get_hst_pi_name(card_dicts, product_id),
        "hst_target_name": get_hst_target_name(card_dicts, product_id),
        "aperture_name": get_aperture_name(
            card_dicts, shm_card_dicts, instrument, product_id
        ),
        "exposure_duration": get_exposure_duration(card_dicts, product_id),
        "exposure_type": get_exposure_type(card_dicts, instrument, product_id),
        "filter_name": get_filter_name(card_dicts, instrument, product_id),
        "fine_guidance_system_lock_type": get_fine_guidance_system_lock_type(
            card_dicts, product_id
        ),
        "instrument_mode_id": get_instrument_mode_id(
            card_dicts, instrument, product_id
        ),
        "moving_target_flag": "true",
    }

    if instrument == "acs":
        parameters_instrument = parameters_acs(
            {
                "detector_id": get_detector_id(card_dicts, instrument, product_id),
                "gain_mode_id": get_gain_mode_id(card_dicts, instrument, product_id),
                "observation_type": get_observation_type(
                    card_dicts, instrument, product_id
                ),
                "repeat_exposure_count": get_repeat_exposure_count(product_id),
                "subarray_flag": get_subarray_flag(card_dicts, instrument, product_id),
            }
        )
    elif instrument == "wfpc2":
        parameters_instrument = parameters_wfpc2(
            {
                "bandwidth": get_bandwidth(card_dicts, instrument, product_id),
                "center_filter_wavelength": get_center_filter_wavelength(
                    card_dicts, instrument, product_id
                ),
                "targeted_detector_id": get_targeted_detector_id(
                    get_aperture_name(
                        card_dicts, shm_card_dicts, instrument, product_id
                    )
                ),
                "gain_mode_id": get_gain_mode_id(card_dicts, instrument, product_id),
            }
        )
    elif instrument == "wfc3":
        parameters_instrument = parameters_wfc3(
            {
                "detector_id": get_detector_id(card_dicts, instrument, product_id),
                "observation_type": get_observation_type(
                    card_dicts, instrument, product_id
                ),
                "repeat_exposure_count": get_repeat_exposure_count(product_id),
                "subarray_flag": get_subarray_flag(card_dicts, instrument, product_id),
            }
        )
    else:
        assert False, "Bad instrument value: %s" % instrument

    return hst(
        {
            "parameters_general": parameters_general(d),
            "parameters_instrument": parameters_instrument,
        }
    )
