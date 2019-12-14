"""
Functionality to build an ``<hst:HST />`` XML element using a SQLite
database.
"""
from typing import TYPE_CHECKING

from pdart.new_labels.HstParametersXml import get_pc1_flag, \
    get_targeted_detector_id, get_wf2_flag, get_wf3_flag, get_wf4_flag, hst, \
    parameters_acs, parameters_general, parameters_wfc3, parameters_wfpc2
from pdart.new_labels.Placeholders import known_placeholder, placeholder, \
    placeholder_float, placeholder_int
from pdart.rules.Combinators import multiple_implementations

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional
    from pdart.xml.Templates import NodeBuilder


def get_repeat_exposure_count(product_id):
    # type: (unicode) -> unicode
    """
    Return a placeholder integer for the ``<repeat_exposure_count
    />`` XML element, noting the problem.
    """
    return placeholder_int(product_id, 'repeat_exposure_count')


def _get_subarray_flag(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    """
    Return placeholder text for the ``<subarray_flag />`` XML element,
    noting the problem.
    """
    if instrument != 'wfpc2':
        return card_dicts[0]['SUBARRAY']
    assert False

def _get_subarray_flag_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder(product_id, 'subarry_flag')

get_subarray_flag = multiple_implementations(
    'get_subarray_flag',
    _get_subarray_flag,
    _get_subarray_flag_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return text for the ``<subarray_flag />`` XML element.
"""


##############################
# get_aperture_name
##############################


def _get_aperture_name(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        try:
            return card_dicts[0]['APERTURE']
        except KeyError:
            return card_dicts[0]['APEROBJ']
    else:
        return card_dicts[0]['APERTURE']


def _get_aperture_name_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder(product_id, 'aperture_name')


get_aperture_name = multiple_implementations(
    'get_aperture_name',
    _get_aperture_name,
    _get_aperture_name_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return text for the ``<aperture_name />`` XML element.
"""


##############################
# get_bandwidth
##############################


def _get_bandwidth(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        bandwid = float(card_dicts[0]['BANDWID'])
        return str(bandwid * 1.0e-4)
    assert False


def _get_bandwidth_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder_float(product_id, 'bandwidth')


get_bandwidth = multiple_implementations(
    'get_bandwidth',
    _get_bandwidth,
    _get_bandwidth_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return a float for the ``<bandwidth />`` XML element.
"""


##############################
# get_center_filter_wavelength
##############################

def _get_center_filter_wavelength(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        centrwv = float(card_dicts[0]['CENTRWV'])
        return str(centrwv * 1.0e-4)
    else:
        raise Exception('Unhandled instrument %s' % instrument)


def _get_center_filter_wavelength_placeholder(card_dicts,
                                              instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder_float(product_id, 'center_filter_wavelength')


get_center_filter_wavelength = multiple_implementations(
    'get_center_filter_wavelength',
    _get_center_filter_wavelength,
    _get_center_filter_wavelength_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return a float for the ``<center_filter_wavelength />`` XML element.
"""


##############################
# get_detector_id
##############################

def _get_detector_id(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    detector = card_dicts[0]['DETECTOR']
    if instrument == 'wfpc2':
        if detector == '1':
            return 'PC1'
        else:
            return 'WF' + detector
    else:
        return detector


def _get_detector_id_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder(product_id, 'detector_id')


get_detector_id = multiple_implementations(
    'get_detector_id',
    _get_detector_id,
    _get_detector_id_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return text for the ``<detector_id />`` XML element.
"""


##############################
# get_exposure_duration
##############################

def _get_exposure_duration(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return str(card_dicts[0]['EXPTIME'])


def _get_exposure_duration_placeholder(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return placeholder_float(product_id, 'exposure_duration')


get_exposure_duration = multiple_implementations(
    'get_exposure_duration',
    _get_exposure_duration,
    _get_exposure_duration_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode], unicode]
"""
Return a float for the ``<exposure_duration />`` XML element.
"""


##############################
# get_exposure_type
##############################

def _get_exposure_type(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], str, str) -> unicode
    if instrument == 'acs':
        try:
            return card_dicts[0]['EXPFLAG']
        except KeyError:
            return 'UNK'
    else:
        return card_dicts[0]['EXPFLAG']


def _get_exposure_type_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], str, str) -> unicode
    return placeholder(product_id, 'exposure_type')


get_exposure_type = multiple_implementations(
    'get_exposure_type',
    _get_exposure_type,
    _get_exposure_type_placeholder
)  # type: Callable[[List[Dict[str, Any]], str, str], unicode]
"""
Return text for the ``<exposure_type />`` XML element.
"""


##############################
# get_filter_name
##############################

def _get_filter_name(card_dicts, instrument, product_name):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        filtnam1 = card_dicts[0]['FILTNAM1'].strip()
        filtnam2 = card_dicts[0]['FILTNAM2'].strip()
        if filtnam1 == '':
            return filtnam2
        elif filtnam2 == '':
            return filtnam1
        else:
            return '%s+%s' % (filtnam1, filtnam2)
    elif instrument == 'acs':
        filter1 = card_dicts[0]['FILTER1'].strip()
        filter2 = card_dicts[0]['FILTER2'].strip()
        if filter1.startswith('CLEAR'):
            if filter2.startswith('CLEAR'):
                return 'CLEAR'
            else:
                return filter2
        else:
            if filter2.startswith('CLEAR'):
                return filter1
            else:
                return '%s+%s' % (filter1, filter2)
    else:
        assert (instrument == 'wfc3')
        return card_dicts[0]['FILTER']


def _get_filter_name_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder(product_id, 'filter_name')


get_filter_name = multiple_implementations(
    'get_filter_name',
    _get_filter_name,
    _get_filter_name_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return text for the ``<filter_name />`` XML element.
"""


##############################
# get_fine_guidance_system_lock_type
##############################

def _get_fine_guidance_system_lock_type(card_dicts, product_id):
    try:
        return card_dicts[0]['FGSLOCK']
    except KeyError:
        return 'UNK'


def _get_fine_guidance_system_lock_type_placeholder(card_dicts, product_id):
    return placeholder(product_id, 'fine_guidance_system_lock_type')


get_fine_guidance_system_lock_type = multiple_implementations(
    'get_fine_guidance_system_lock_type',
    _get_fine_guidance_system_lock_type,
    _get_fine_guidance_system_lock_type_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode], unicode]
"""
Return text for the ``<fine_guidance_system_lock_type />`` XML element.
"""


##############################
# get_gain_mode_id
##############################


def _get_gain_mode_id(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    try:
        atodgain = card_dicts[0]['ATODGAIN']
    except KeyError:
        return 'N/A'
    if instrument == 'acs':
        return str(atodgain)
    elif instrument == 'wfpc2':
        return 'A2D' + str(int(atodgain))
    assert False


def _get_gain_mode_id_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder(product_id, 'gain_mode_id')


get_gain_mode_id = multiple_implementations(
    'get_gain_mode_id',
    _get_gain_mode_id,
    _get_gain_mode_id_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return text for the ``<gain_mode_id />`` XML element.
"""


##############################
# get_gyroscope_mode
##############################

def _get_gyroscope_mode(card_dicts, product_id):
    return card_dicts[0]['GYROMODE']

def _get_gyroscope_mode_placeholder(card_dicts, product_id):
    return placeholder(product_id, 'gyroscope_mode')

get_gyroscope_mode  = multiple_implementations(
    'get_gyroscope_mode',
    _get_gyroscope_mode,
    _get_gyroscope_mode_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode], unicode]


##############################
# get_hst_pi_name
##############################

def _get_hst_pi_name(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    pr_inv_l = card_dicts[0]['PR_INV_L']
    pr_inv_f = card_dicts[0]['PR_INV_F']
    try:
        pr_inv_m = card_dicts[0]['PR_INV_M']
        return '%s, %s %s' % (pr_inv_l, pr_inv_f, pr_inv_m)
    except KeyError:
        return '%s, %s' % (pr_inv_l, pr_inv_f)


def _get_hst_pi_name_placeholder(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return placeholder(product_id, 'hst_pi_name')


get_hst_pi_name = multiple_implementations(
    'get_hst_pi_name',
    _get_hst_pi_name,
    _get_hst_pi_name_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode], unicode]
"""
Return text for the ``<hst_pi_name />`` XML element.
"""


##############################
# get_hst_proposal_id
##############################

def _get_hst_proposal_id(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return str(card_dicts[0]['PROPOSID'])


def _get_hst_proposal_id_placeholder(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return placeholder_int(product_id, 'hst_proposal_id')


get_hst_proposal_id = multiple_implementations(
    'get_hst_proposal_id',
    _get_hst_proposal_id,
    _get_hst_proposal_id_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode], unicode]
"""
Return text for the ``<hst_proposal_id />`` XML element.
"""


##############################
# get_hst_target_name
##############################

def _get_hst_target_name(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return card_dicts[0]['TARGNAME']


def _get_hst_target_name_placeholder(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return placeholder(product_id, 'hst_target_name')


get_hst_target_name = multiple_implementations(
    'get_hst_target_name',
    _get_hst_target_name,
    _get_hst_target_name_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode], unicode]
"""
Return text for the ``<hst_target_name />`` XML element.
"""


##############################
# get_instrument_mode_id
##############################

def _get_instrument_mode_id(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    if instrument == 'acs':
        try:
            return card_dicts[0]['OBSMODE']
        except KeyError:
            return 'UNK'
    if instrument == 'wfpc2':
        return card_dicts[0]['MODE']
    else:
        assert (instrument == 'wfc3')
        return card_dicts[0]['OBSMODE']


def _get_instrument_mode_id_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder(product_id, 'instrument_mode_id')


get_instrument_mode_id = multiple_implementations(
    'get_instrument_mode_id',
    _get_instrument_mode_id,
    _get_instrument_mode_id_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return text for the ``<instrument_mode_id />`` XML element.
"""


##############################
# get_observation_type
##############################

def _get_observation_type(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    if instrument != 'wfpc2':
        return card_dicts[0]['OBSTYPE']
    else:
        raise Exception('Unhandled instrument %s' % instrument)


def _get_observation_type_placeholder(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], unicode, unicode) -> unicode
    return placeholder(product_id, 'observation_type')


get_observation_type = multiple_implementations(
    'get_observation_type',
    _get_observation_type,
    _get_observation_type_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode, unicode], unicode]
"""
Return text for the ``<observation_type />`` XML element.
"""

##############################
# get_stsci_group_id: NOTE the name is going to change
##############################

def _get_stsci_group_id(card_dicts, product_id):
    try:
        asn_id = card_dicts[0]['ASN_ID']
        if asn_id == 'NONE':
            return card_dicts[0]['ROOTNAME']
        else:
            return asn_id
    except KeyError:
        return card_dicts[0]['ROOTNAME']


def _get_stsci_group_id_placeholder(card_dicts, product_id):
    # type: (List[Dict[str, Any]], unicode) -> unicode
    return placeholder(product_id, 'stsci_group_id')

get_stsci_group_id = multiple_implementations(
    'get_stsci_group_id',
    _get_stsci_group_id,
    _get_stsci_group_id_placeholder
)  # type: Callable[[List[Dict[str, Any]], unicode], unicode]
"""
Return text for the ``<stsci_group_id />`` XML element.
"""

##############################

def get_hst_parameters(card_dicts, instrument, product_id):
    # type: (List[Dict[str, Any]], str, str) -> NodeBuilder
    """Return an ``<hst:HST />`` XML element."""
    d = {'stsci_group_id': get_stsci_group_id(card_dicts, product_id),
         'hst_proposal_id': get_hst_proposal_id(card_dicts, product_id),
         'hst_pi_name': get_hst_pi_name(card_dicts, product_id),
         'hst_target_name': get_hst_target_name(card_dicts, product_id),
         'aperture_name': get_aperture_name(card_dicts, instrument,
                                            product_id),
         'exposure_duration': get_exposure_duration(card_dicts, product_id),
         'exposure_type':
             get_exposure_type(card_dicts, instrument, product_id),
         'filter_name': get_filter_name(card_dicts, instrument, product_id),
         'fine_guidance_system_lock_type':
             get_fine_guidance_system_lock_type(card_dicts, product_id),
         'gyroscope_mode': get_gyroscope_mode(card_dicts, product_id),
         'instrument_mode_id': get_instrument_mode_id(card_dicts,
                                                      instrument,
                                                      product_id),
         'moving_target_flag': 'true'}

    if instrument == 'acs':
        parameters_instrument = parameters_acs(
            {'detector_id': get_detector_id(card_dicts, instrument,
                                            product_id),
             'gain_mode_id': get_gain_mode_id(card_dicts, instrument,
                                              product_id),
             'observation_type':
                 get_observation_type(card_dicts, instrument, product_id),
             'repeat_exposure_count': get_repeat_exposure_count(product_id),
             'subarray_flag':
                 get_subarray_flag(card_dicts, instrument, product_id)})
    elif instrument == 'wfpc2':
        parameters_instrument = parameters_wfpc2(
            {'bandwidth': get_bandwidth(card_dicts, instrument, product_id),
             'center_filter_wavelength':
                 get_center_filter_wavelength(card_dicts,
                                              instrument,
                                              product_id),
             'targeted_detector_id':
                 get_targeted_detector_id(card_dicts[0]['APERTURE']),
             'gain_mode_id': get_gain_mode_id(card_dicts, instrument,
                                              product_id),
             'pc1_flag': get_pc1_flag(product_id, instrument),
             'wf2_flag': get_wf2_flag(product_id, instrument),
             'wf3_flag': get_wf3_flag(product_id, instrument),
             'wf4_flag': get_wf4_flag(product_id, instrument)})
    elif instrument == 'wfc3':
        parameters_instrument = parameters_wfc3(
            {'detector_id': get_detector_id(card_dicts,
                                            instrument,
                                            product_id),
             'observation_type':
                 get_observation_type(card_dicts, instrument, product_id),
             'repeat_exposure_count':
                 get_repeat_exposure_count(product_id),
             'subarray_flag':
                 get_subarray_flag(card_dicts, instrument, product_id)})
    else:
        assert False, 'Bad instrument value: %s' % instrument

    return hst({
        'parameters_general': parameters_general(d),
        'parameters_instrument': parameters_instrument})
