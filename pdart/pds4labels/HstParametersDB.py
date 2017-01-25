"""
Functionality to build an ``<hst:HST />`` XML element using a SQLite
database.
"""
from pdart.pds4labels.HstParametersXml import *
from pdart.rules.Combinators import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pdart.pds4labels.DBCalls import Headers


def get_db_repeat_exposure_count(product_id):
    # type: (unicode) -> unicode
    """
    Return a placeholder integer for the ``<repeat_exposure_count
    />`` XML element, noting the problem.
    """
    return placeholder_int(product_id, 'repeat_exposure_count')


def get_db_subarray_flag(product_id):
    # type: (unicode) -> unicode
    """
    Return placeholder text for the ``<subarray_flag />`` XML element,
    noting the problem.
    """
    return placeholder(product_id, 'subarray_flag')

##############################
# get_db_aperture_type
##############################


def _get_db_aperture_type(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        return _get_db_aperture_type_placeholder(headers, instrument,
                                                 product_id)
    else:
        return headers[0]['APERTURE']


def _get_db_aperture_type_placeholder(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder(product_id, 'aperture_type')


get_db_aperture_type = multiple_implementations(
    'get_db_aperture_type',
    _get_db_aperture_type,
    _get_db_aperture_type_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return text for the ``<aperture_type />`` XML element.
"""

##############################
# get_db_bandwidth
##############################


def _get_db_bandwidth(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        bandwid = float(headers[0]['BANDWID'])
        return str(bandwid * 1.e-4)


def _get_db_bandwidth_placeholder(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder_float(product_id, 'bandwidth')

get_db_bandwidth = multiple_implementations(
    'get_db_bandwidth',
    _get_db_bandwidth,
    _get_db_bandwidth_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return a float for the ``<bandwidth />`` XML element.
"""


##############################
# get_db_center_filter_wavelength
##############################

def _get_db_center_filter_wavelength(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        centrwv = float(headers[0]['CENTRWV'])
        return str(centrwv * 1.e-4)
    else:
        raise Exception('Unhandled instrument %s' % instrument)


def _get_db_center_filter_wavelength_placeholder(headers,
                                                 instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder_float(product_id, 'center_filter_wavelength')

get_db_center_filter_wavelength = multiple_implementations(
    'get_db_center_filter_wavelength',
    _get_db_center_filter_wavelength,
    _get_db_center_filter_wavelength_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return a float for the ``<center_filter_wavelength />`` XML element.
"""


##############################
# get_db_detector_id
##############################

def _get_db_detector_id(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    detector = headers[0]['DETECTOR']
    if instrument == 'wfpc2':
        if detector == '1':
            return 'PC1'
        else:
            return 'WF' + detector
    else:
        return detector


def _get_db_detector_id_placeholder(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder(product_id, 'detector_id')

get_db_detector_id = multiple_implementations(
    'get_db_detector_id',
    _get_db_detector_id,
    _get_db_detector_id_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return text for the ``<detector_id />`` XML element.
"""


##############################
# get_db_exposure_duration
##############################

def _get_db_exposure_duration(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return str(headers[0]['EXPTIME'])


def _get_db_exposure_duration_placeholder(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return placeholder_float(product_id, 'exposure_duration')

get_db_exposure_duration = multiple_implementations(
    'get_db_exposure_duration',
    _get_db_exposure_duration,
    _get_db_exposure_duration_placeholder)
# type: Callable[[Headers, unicode], unicode]
"""
Return a float for the ``<exposure_duration />`` XML element.
"""


##############################
# get_db_exposure_type
##############################

def _get_db_exposure_type(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return headers[0]['EXPFLAG']


def _get_db_exposure_type_placeholder(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return placeholder(product_id, 'exposure_type')

get_db_exposure_type = multiple_implementations(
    'get_db_exposure_type',
    _get_db_exposure_type,
    _get_db_exposure_type_placeholder)
# type: Callable[[Headers, unicode], unicode]
"""
Return text for the ``<exposure_type />`` XML element.
"""


##############################
# get_db_filter_name
##############################

def _get_db_filter_name(headers, instrument, product_name):
    # type: (Headers, unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        filtnam1 = headers[0]['FILTNAM1'].strip()
        filtnam2 = headers[0]['FILTNAM2'].strip()
        if filtnam1 == '':
            return filtnam2
        elif filtnam2 == '':
            return filtnam1
        else:
            return '%s+%s' % (filtnam1, filtnam2)
    elif instrument == 'acs':
        filter1 = headers[0]['FILTER1']
        filter2 = headers[0]['FILTER2']
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
    elif instrument == 'wfc3':
        return headers[0]['FILTER']


def _get_db_filter_name_placeholder(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder(product_id, 'filter_name')

get_db_filter_name = multiple_implementations(
    'get_db_filter_name',
    _get_db_filter_name,
    _get_db_filter_name_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return text for the ``<filter_name />`` XML element.
"""


##############################
# get_db_fine_guidance_system_lock_type
##############################

def _get_db_fine_guidance_system_lock_type(headers, product_id):
    return headers[0]['FGSLOCK']


def _get_db_fine_guidance_system_lock_type_placeholder(headers, product_id):
    return placeholder(product_id, 'fine_guidance_system_lock_type')

get_db_fine_guidance_system_lock_type = multiple_implementations(
    'get_db_fine_guidance_system_lock_type',
    _get_db_fine_guidance_system_lock_type,
    _get_db_fine_guidance_system_lock_type_placeholder)
# type: Callable[[Headers, unicode], unicode]
"""
Return text for the ``<fine_guidance_system_lock_type />`` XML element.
"""

##############################
# get_db_gain_mode_id
##############################


def _get_db_gain_mode_id(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    atodgain = headers[0]['ATODGAIN']
    if instrument == 'acs':
        return str(atodgain)
    elif instrument == 'wfpc2':
        return 'A2D' + str(int(atodgain))


def _get_db_gain_mode_id_placeholder(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder(product_id, 'gain_mode_id')

get_db_gain_mode_id = multiple_implementations(
    'get_db_gain_mode_id',
    _get_db_gain_mode_id,
    _get_db_gain_mode_id_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return text for the ``<gain_mode_id />`` XML element.
"""


##############################
# get_db_hst_pi_name
##############################

def _get_db_hst_pi_name(headers, product_id):
    # type: (Headers, unicode) -> unicode
    pr_inv_l = headers[0]['PR_INV_L']
    pr_inv_f = headers[0]['PR_INV_F']
    pr_inv_m = headers[0]['PR_INV_M']
    return '%s, %s %s' % (pr_inv_l, pr_inv_f, pr_inv_m)


def _get_db_hst_pi_name_placeholder(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return placeholder(product_id, 'hst_pi_name')

get_db_hst_pi_name = multiple_implementations(
    'get_db_hst_pi_name',
    _get_db_hst_pi_name,
    _get_db_hst_pi_name_placeholder)
# type: Callable[[Headers, unicode], unicode]
"""
Return text for the ``<hst_pi_name />`` XML element.
"""


##############################
# get_db_hst_proposal_id
##############################

def _get_db_hst_proposal_id(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return str(headers[0]['PROPOSID'])


def _get_db_hst_proposal_id_placeholder(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return placeholder_int(product_id, 'hst_proposal_id')


get_db_hst_proposal_id = multiple_implementations(
    'get_db_hst_proposal_id',
    _get_db_hst_proposal_id,
    _get_db_hst_proposal_id_placeholder)
# type: Callable[[Headers, unicode], unicode]
"""
Return text for the ``<hst_proposal_id />`` XML element.
"""


##############################
# get_db_hst_target_name
##############################

def _get_db_hst_target_name(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return headers[0]['TARGNAME']


def _get_db_hst_target_name_placeholder(headers, product_id):
    # type: (Headers, unicode) -> unicode
    return placeholder(product_id, 'hst_target_name')


get_db_hst_target_name = multiple_implementations(
    'get_db_hst_target_name',
    _get_db_hst_target_name,
    _get_db_hst_target_name_placeholder)
# type: Callable[[Headers, unicode], unicode]
"""
Return text for the ``<hst_target_name />`` XML element.
"""


##############################
# get_db_instrument_mode_id
##############################

def _get_db_instrument_mode_id(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    if instrument == 'wfpc2':
        return headers[0]['MODE']
    else:
        return headers[0]['OBSMODE']


def _get_db_instrument_mode_id_placeholder(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder(product_id, 'instrument_mode_id')

get_db_instrument_mode_id = multiple_implementations(
    'get_db_instrument_mode_id',
    _get_db_instrument_mode_id,
    _get_db_instrument_mode_id_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return text for the ``<instrument_mode_id />`` XML element.
"""


##############################
# get_db_observation_type
##############################

def _get_db_observation_type(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    if instrument != 'wfpc2':
        return headers[0]['OBSTYPE']
    else:
        raise Exception('Unhandled instrument %s' % instrument)


def _get_db_observation_type_placeholder(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> unicode
    return placeholder(product_id, 'observation_type')


get_db_observation_type = multiple_implementations(
    'get_db_observation_type',
    _get_db_observation_type,
    _get_db_observation_type_placeholder)
# type: Callable[[Headers, unicode, unicode], unicode]
"""
Return text for the ``<observation_type />`` XML element.
"""


##############################

def get_db_hst_parameters(headers, instrument, product_id):
    # type: (Headers, unicode, unicode) -> NodeBuilder
    """Return an ``<hst:HST />`` XML element."""
    d = {'stsci_group_id': known_placeholder(product_id, 'stsci_group_id'),
         'hst_proposal_id': get_db_hst_proposal_id(headers, product_id),
         'hst_pi_name': get_db_hst_pi_name(headers, product_id),
         'hst_target_name': get_db_hst_target_name(headers, product_id),
         'aperture_type': get_db_aperture_type(headers, instrument,
                                               product_id),
         'exposure_duration': get_db_exposure_duration(headers, product_id),
         'exposure_type': get_db_exposure_type(headers, product_id),
         'filter_name': get_db_filter_name(headers, instrument, product_id),
         'fine_guidance_system_lock_type':
             get_db_fine_guidance_system_lock_type(headers, product_id),
         'gyroscope_mode': known_placeholder(product_id,
                                             'gyroscope_mode'),
         'instrument_mode_id': get_db_instrument_mode_id(headers,
                                                         instrument,
                                                         product_id),
         'moving_target_flag': 'true'}

    if instrument == 'acs':
        parameters_instrument = parameters_acs(
            {'detector_id': get_db_detector_id(headers, instrument,
                                               product_id),
             'gain_mode_id': get_db_gain_mode_id(headers, instrument,
                                                 product_id),
             'observation_type':
                 get_db_observation_type(headers, instrument, product_id),
             'repeat_exposure_count': get_db_repeat_exposure_count(product_id),
             'subarray_flag': get_db_subarray_flag(product_id)})
    elif instrument == 'wfpc2':
        header = None
        # type: unicode
        parameters_instrument = parameters_wfpc2(
            {'bandwidth': get_db_bandwidth(headers, instrument, product_id),
             'center_filter_wavelength':
                 get_db_center_filter_wavelength(headers,
                                                 instrument, product_id),
             'targeted_detector_id':
                 get_targeted_detector_id(product_id, instrument,
                                          header),
             'gain_mode_id': get_db_gain_mode_id(headers, instrument,
                                                 product_id),
             'pc1_flag': get_pc1_flag(product_id, instrument,
                                      header),
             'wf2_flag': get_wf2_flag(product_id, instrument,
                                      header),
             'wf3_flag': get_wf3_flag(product_id, instrument,
                                      header),
             'wf4_flag': get_wf4_flag(product_id, instrument,
                                      header)})
    elif instrument == 'wfc3':
        parameters_instrument = parameters_wfc3(
            {'detector_id': get_db_detector_id(headers,
                                               instrument,
                                               product_id),
             'observation_type':
                 get_db_observation_type(headers, instrument, product_id),
             'repeat_exposure_count':
                 get_db_repeat_exposure_count(product_id),
             'subarray_flag':
                 get_db_subarray_flag(product_id)})
    else:
        assert False, 'Bad instrument value: %s' % instrument

    return hst({
            'parameters_general': parameters_general(d),
            'parameters_instrument': parameters_instrument})
