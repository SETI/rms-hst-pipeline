from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


hst = interpret_template("""<hst:HST>
<NODE name="parameters_general"/>
<NODE name="parameters_instrument"/>
</hst:HST>""")

parameters_general = interpret_template("""<hst:Parameters_General>
  <hst:stsci_group_id><NODE name="stsci_group_id" /></hst:stsci_group_id>
  <hst:hst_proposal_id><NODE name="hst_proposal_id" /></hst:hst_proposal_id>
  <hst:hst_pi_name><NODE name="hst_pi_name" /></hst:hst_pi_name>
  <hst:hst_target_name><NODE name="hst_target_name" /></hst:hst_target_name>
  <hst:aperture_type><NODE name="aperture_type" /></hst:aperture_type>
  <hst:exposure_duration><NODE name="exposure_duration" />\
</hst:exposure_duration>
  <hst:exposure_type><NODE name="exposure_type" /></hst:exposure_type>
  <hst:filter_name><NODE name="filter_name" /></hst:filter_name>
  <hst:fine_guidance_system_lock_type>\
<NODE name="fine_guidance_system_lock_type" />\
</hst:fine_guidance_system_lock_type>
  <hst:gyroscope_mode><NODE name="gyroscope_mode" /></hst:gyroscope_mode>
  <hst:instrument_mode_id><NODE name="instrument_mode_id" />\
</hst:instrument_mode_id>
  <hst:moving_target_flag><NODE name="moving_target_flag" />\
</hst:moving_target_flag>
</hst:Parameters_General>""")

parameters_acs = interpret_template("""<hst:Parameters_ACS>
<hst:detector_id><NODE name="detector_id" /></hst:detector_id>
<hst:gain_mode_id><NODE name="gain_mode_id" /></hst:gain_mode_id>
<hst:observation_type><NODE name="observation_type" /></hst:observation_type>
<hst:repeat_exposure_count><NODE name="repeat_exposure_count" />\
</hst:repeat_exposure_count>
<hst:subarray_flag><NODE name="subarray_flag" /></hst:subarray_flag>
</hst:Parameters_ACS>""")

parameters_wfc3 = interpret_template("""<hst:Parameters_WFC3>
</hst:Parameters_WFC3>""")

parameters_wfpc2 = interpret_template("""<hst:Parameters_WFPC2>
<hst:bandwidth><NODE name="bandwidth" /></hst:bandwidth>
<hst:center_filter_wavelength><NODE name="center_filter_wavelength" />\
</hst:center_filter_wavelength>
<hst:targeted_detector_id><NODE name="targeted_detector_id" />\
</hst:targeted_detector_id>
<hst:gain_mode_id><NODE name="gain_mode_id" /></hst:gain_mode_id>
<hst:pc1_flag><NODE name="pc1_flag" /></hst:pc1_flag>
<hst:wf2_flag><NODE name="wf2_flag" /></hst:wf2_flag>
<hst:wf3_flag><NODE name="wf3_flag" /></hst:wf3_flag>
<hst:wf4_flag><NODE name="wf4_flag" /></hst:wf4_flag>
</hst:Parameters_WFPC2>""")

wrapper = interpret_document_template("""<NODE name="wrapped" />""")


def get_repeat_exposure_count(instrument, header):
    return placeholder('repeat_exposure_count')


def get_subarray_flag(instrument, header):
    return placeholder('subarray_flag')


def get_targeted_detector_id(instrument, header):
    return placeholder('targeted_detector_id')


def get_pc1_flag(instrument, header):
    # return placeholder('pc1_flag')
    return '0'


def get_wf2_flag(instrument, header):
    # return placeholder('wf2_flag')
    return '0'


def get_wf3_flag(instrument, header):
    # return placeholder('wf3_flag')
    return '0'


def get_wf4_flag(instrument, header):
    # return placeholder('wf4_flag')
    return '0'


def get_aperture_type(instrument, header):
    if instrument == 'wfpc2':
        # TODO: should be None?  But it's required.  What to do?
        return placeholder('aperature_type')
    else:
        return header['APERTURE']


def get_bandwidth(instrument, header):
    if instrument == 'wfpc2':
        try:
            return str(header['BANDWID'] * 1.e-4)
        except KeyError:
            # return placeholder('bandwidth')
            return '0.0'


def get_center_filter_wavelength(instrument, header):
    if instrument == 'wfpc2':
        try:
            return str(header['CENTRWV'] * 1.e-4)
        except KeyError:
            # return placeholder('center_filter_wavelength')
            return '0.0'


def get_detector_id(instrument, header):
    detector = header['DETECTOR']
    if instrument == 'wfpc2':
        if detector == '1':
            return 'PC1'
        else:
            return 'WF' + detector
    else:
        return detector


def get_exposure_duration(instrument, header):
    try:
        return str(header['EXPTIME'])
    except KeyError:
        # return placeholder('exposure_duration')
        return '0.0'


def get_exposure_type(instrument, header):
    try:
        return str(header['EXPFLAG'])
    except KeyError:
        return placeholder('exposure_type')


def get_filter_name(instrument, header):
    try:
        if instrument == 'wfpc2':
            filtnam1 = header['FILTNAM1'].strip()
            filtnam2 = header['FILTNAM2'].strip()
            if filtnam1 == '':
                return filtnam2
            elif filtnam2 == '':
                return filtnam1
            else:
                return '%s+%s' % (filtnam1, filtnam2)
        elif instrument == 'acs':
            filter1 = header['FILTER1']
            filter2 = header['FILTER2']
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
            return header['FILTER']
    except KeyError:
        return placeholder('filter_name')


def get_fine_guidance_system_lock_type(instrument, header):
    try:
        return header['FSGLOCK']
    except KeyError:
        return placeholder('fine_guidance_system_lock_type')


def get_gain_mode_id(instrument, header):
    if instrument == 'acs':
        return str(header['ATODGAIN'])
    elif instrument == 'wfpc2':
        try:
            return 'A2D' + str(int(header['ATODGAIN']))
        except KeyError:
            return placeholder('gain_mode_id')


def get_hst_pi_name(instrument, header):
    try:
        return '%s, %s %s' % (header['PR_INV_L'],
                              header['PR_INV_F'],
                              header['PR_INV_M'])
    except KeyError:
        return placeholder('hst_pi_name')


def get_hst_proposal_id(instrument, header):
    try:
        return str(header['PROPOSID'])
    except KeyError:
        return placeholder('hst_proposal_id')


def get_hst_target_name(instrument, header):
    try:
        return header['TARGNAME']
    except KeyError:
        return placeholder('hst_target_name')


def get_instrument_mode_id(instrument, header):
    try:
        if instrument == 'wfpc2':
            return header['MODE']
        else:
            return header['OBSMODE']
    except KeyError:
        return placeholder('instrument_mode_id')


def get_observation_type(instrument, header):
    if instrument == 'wfpc2':
        return None
    else:
        return header['OBSTYPE']


def placeholder(tag):
    return '### placeholder for %s ###' % tag


class HstParametersReduction(Reduction):
    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # return (Doc -> Node)
        res = get_reduced_fits_files()[0]
        assert res
        return res

    def reduce_fits_file(self, file, get_reduced_hdus):
        # returns (Doc -> Node)
        res = get_reduced_hdus()[0]
        assert res
        return res

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # returns (Doc -> Node) or None
        if n == 0:
            instrument = 'wfpc2'
            header = hdu.header

            d = {'stsci_group_id': placeholder('stsci_group_id'),
                 'hst_proposal_id': get_hst_proposal_id(instrument, header),
                 'hst_pi_name': get_hst_pi_name(instrument, header),
                 'hst_target_name': get_hst_target_name(instrument, header),
                 'aperture_type': get_aperture_type(instrument, header),
                 'exposure_duration': get_exposure_duration(instrument,
                                                            header),
                 'exposure_type': get_exposure_type(instrument, header),
                 'filter_name': get_filter_name(instrument, header),

                 'fine_guidance_system_lock_type':
                     get_fine_guidance_system_lock_type(instrument, header),
                 'gyroscope_mode': placeholder('gyroscope_mode'),
                 'instrument_mode_id': get_instrument_mode_id(instrument,
                                                              header),
                 'moving_target_flag': 'true'}

#                (instrument_id, 'instrument_id',
#                 lambda: header['INSTRUME']),
#                (detector_id, 'detector_id',
#                 lambda: get_detector_id(instrument, header)),
#                (observation_type, 'observation_type',
#                 lambda: get_observation_type(instrument,
#                                              header)),
#                (product_type, 'product_type',
#                 lambda: placeholder('product_type')),
#                (center_filter_wavelength, 'center_filter_wavelength',
#                 lambda: get_center_filter_wavelength(instrument, header)),
#                (bandwidth, 'bandwidth',
#                 lambda: get_bandwidth(instrument, header)),
#                (wavelength_resolution, 'wavelength_resolution',
#                 lambda: placeholder('wavelength_resolution')),
#                (maximum_wavelength, 'maximum_wavelength',
#                 lambda: placeholder('maximum_wavelength')),
#                (minimum_wavelength, 'minimum_wavelength',
#                 lambda: placeholder('minimum_wavelength')),
#                (gain_mode_id, 'gain_mode_id',
#                 lambda: get_gain_mode_id(instrument, header)),
#                (lines, 'lines', lambda: placeholder('lines')),
#                (line_samples, 'line_samples',
#                 lambda: placeholder('line_samples'))

            if instrument == 'acs':
                parameters_instrument = parameters_acs(
                    {'detector_id': get_detector_id(instrument, header),
                     'gain_mode_id': get_gain_mode_id(instrument, header),
                     'observation_type':
                         get_observation_type(instrument, header),
                     'repeat_exposure_count':
                         get_repeat_exposure_count(instrument, header),
                     'subarray_flag': get_subarray_flag(instrument, header)
                     })
            elif instrument == 'wfpc2':
                parameters_instrument = parameters_wfpc2(
                    {'bandwidth': get_bandwidth(instrument, header),
                     'center_filter_wavelength':
                         get_center_filter_wavelength(instrument, header),
                     'targeted_detector_id':
                         get_targeted_detector_id(instrument, header),
                     'gain_mode_id': get_gain_mode_id(instrument, header),
                     'pc1_flag': get_pc1_flag(instrument, header),
                     'wf2_flag': get_wf2_flag(instrument, header),
                     'wf3_flag': get_wf3_flag(instrument, header),
                     'wf4_flag': get_wf4_flag(instrument, header)})
            elif instrument == 'wfc3':
                parameters_instrument = parameters_wfc3({})

            # Wrap the fragment and return it.
            return hst({
                    'parameters_general': parameters_general(d),
                    'parameters_instrument': parameters_instrument})
