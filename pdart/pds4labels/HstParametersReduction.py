from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


hst_parameters = interpret_template("""<General_HST_Parameters>
<FRAGMENT name="hst_parameters"/>
</General_HST_Parameters>""")

stsci_group_id = interpret_template(
    """<stsci_group_id><NODE name="stsci_group_id" /></stsci_group_id>""")

hst_proposal_id = interpret_template(
    """<hst_proposal_id><NODE name="hst_proposal_id" /></hst_proposal_id>""")

hst_pi_name = interpret_template(
    """<hst_pi_name><NODE name="hst_pi_name" /></hst_pi_name>""")

instrument_id = interpret_template(
    """<instrument_id><NODE name="instrument_id" /></instrument_id>""")

detector_id = interpret_template(
    """<detector_id><NODE name="detector_id" /></detector_id>""")

observation_type = interpret_template(
    """<observation_type><NODE name="observation_type" />\
</observation_type>""")

product_type = interpret_template(
    """<product_type><NODE name="product_type" /></product_type>""")

exposure_duration = interpret_template(
    """<exposure_duration><NODE name="exposure_duration" />\
</exposure_duration>""")

hst_target_name = interpret_template(
    """<hst_target_name><NODE name="hst_target_name" /></hst_target_name>""")

filter_name = interpret_template(
    """<filter_name><NODE name="filter_name" /></filter_name>""")

center_filter_wavelength = interpret_template(
    """<center_filter_wavelength><NODE name="center_filter_wavelength" />\
</center_filter_wavelength>""")

bandwidth = interpret_template(
    """<bandwidth><NODE name="bandwidth" /></bandwidth>""")

wavelength_resolution = interpret_template(
    """<wavelength_resolution><NODE name="wavelength_resolution" />\
</wavelength_resolution>""")

maximum_wavelength = interpret_template(
    """<maximum_wavelength><NODE name="maximum_wavelength" />\
</maximum_wavelength>""")

minimum_wavelength = interpret_template(
    """<minimum_wavelength><NODE name="minimum_wavelength" />\
</minimum_wavelength>""")

aperture_type = interpret_template(
    """<aperture_type><NODE name="aperture_type" /></aperture_type>""")

exposure_type = interpret_template(
    """<exposure_type><NODE name="exposure_type" /></exposure_type>""")

fine_guidance_system_lock_type = interpret_template(
    """<fine_guidance_system_lock_type>\
<NODE name="fine_guidance_system_lock_type" />\
</fine_guidance_system_lock_type>""")

gain_mode_id = interpret_template(
    """<gain_mode_id><NODE name="gain_mode_id" /></gain_mode_id>""")

instrument_mode_id = interpret_template(
    """<instrument_mode_id><NODE name="instrument_mode_id" />\
</instrument_mode_id>""")

lines = interpret_template(
    """<lines><NODE name="lines" /></lines>""")

line_samples = interpret_template(
    """<line_samples><NODE name="line_samples" /></line_samples>""")

gyroscope_mode = interpret_template(
    """<gyroscope_mode><NODE name="gyroscope_mode" /></gyroscope_mode>""")

moving_target_flag = interpret_template(
    """<moving_target_flag><NODE name="moving_target_flag" />\
</moving_target_flag>""")


wrapper = interpret_document_template("""<NODE name="wrapped" />""")


def get_aperture_type(instrument, header):
    if instrument == 'wfpc2':
        return None
    else:
        return header['APERTURE']


def get_bandwidth(instrument, header):
    if instrument == 'wfpc2':
        return str(header['BANDWID'] * 1.e-4)


def get_center_filter_wavelength(instrument, header):
    if instrument == 'wfpc2':
        return str(header['CENTRWV'] * 1.e-4)


def get_detector_id(instrument, header):
    detector = header['DETECTOR']
    if instrument == 'wfpc2':
        if detector == '1':
            return 'PC1'
        else:
            return 'WF' + detector
    else:
        return detector


def get_filter_name(instrument, header):
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


def get_gain_mode_id(instrument, header):
    if instrument == 'acs':
        return str(header['ATODGAIN'])
    elif instrument == 'wfpc2':
        return 'A2D' + str(int(header['ATODGAIN']))


def get_hst_pi_name(instrument, header):
    return '%s, %s %s' % (header['PR_INV_L'],
                          header['PR_INV_F'],
                          header['PR_INV_M'])


def get_instrument_mode_id(instrument, header):
    if instrument == 'wfpc2':
        return header['MODE']
    else:
        return header['OBSMODE']


def get_observation_type(instrument, header):
    if instrument == 'wfpc2':
        return None
    else:
        return header['OBSTYPE']


def placeholder(tag):
    return '### placeholder for %s ###' % tag


class HstParametersReduction(Reduction):
    def reduce_product(self, archive, lid, get_reduced_fits_files):
        # return String
        wrapped = get_reduced_fits_files()[0]
        return wrapper({'wrapped': wrapped}).toxml()

    def reduce_fits_file(self, file, get_reduced_hdus):
        # returns (Doc -> Node)
        return get_reduced_hdus()[0]

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # returns (Doc -> Node) or None
        if n == 0:
            instrument = 'acs'
            header = hdu.header

            # This is a list of XML templates, element names, and
            # thunks to calculate what will go inside the templates.
            # For each tuple, we run the thunk and if it returns a
            # non-None answer, we use it as parameter (with the
            # element name) to the XML template.  Each tuple may
            # create a node (function).  We combine them into a
            # fragment (function) and wrap the fragment as our
            # General_HST_Parameters.
            nodeCalcs = [
                (stsci_group_id, 'stsci_group_id',
                 lambda: placeholder('stsci_group_id')),
                (hst_proposal_id, 'hst_proposal_id',
                 lambda: str(header['PROPOSID'])),
                (hst_pi_name, 'hst_pi_name',
                 lambda: get_hst_pi_name(instrument, header)),
                (instrument_id, 'instrument_id',
                 lambda: header['INSTRUME']),
                (detector_id, 'detector_id',
                 lambda: get_detector_id(instrument, header)),
                (observation_type, 'observation_type',
                 lambda: get_observation_type(instrument,
                                              header)),
                (product_type, 'product_type',
                 lambda: placeholder('product_type')),
                (exposure_duration, 'exposure_duration',
                 lambda: str(header['EXPTIME'])),
                (hst_target_name, 'hst_target_name',
                 lambda: header['TARGNAME']),
                (filter_name, 'filter_name',
                 lambda: get_filter_name(instrument, header)),
                (center_filter_wavelength, 'center_filter_wavelength',
                 lambda: get_center_filter_wavelength(instrument, header)),
                (bandwidth, 'bandwidth',
                 lambda: get_bandwidth(instrument, header)),
                (wavelength_resolution, 'wavelength_resolution',
                 lambda: placeholder('wavelength_resolution')),
                (maximum_wavelength, 'maximum_wavelength',
                 lambda: placeholder('maximum_wavelength')),
                (minimum_wavelength, 'minimum_wavelength',
                 lambda: placeholder('minimum_wavelength')),
                (aperture_type, 'aperture_type',
                 lambda: get_aperture_type(instrument, header)),
                (exposure_type, 'exposure_type', lambda: header['EXPFLAG']),
                (fine_guidance_system_lock_type,
                 'fine_guidance_system_lock_type', lambda: header['FSGLOCK']),
                (gain_mode_id, 'gain_mode_id',
                 lambda: get_gain_mode_id(instrument, header)),
                (instrument_mode_id, 'instrument_mode_id',
                 lambda: header['OBSMODE']),
                (lines, 'lines', lambda: placeholder('lines')),
                (line_samples, 'line_samples',
                 lambda: placeholder('line_samples')),
                (gyroscope_mode, 'gyroscope_mode',
                 lambda: placeholder('gyroscope_mode')),
                (moving_target_flag, 'moving_target_flag',
                 lambda: placeholder('moving_target_flag'))
                ]

            # Turn each (successful) tuple into a node.
            nodes = []
            for (template, name, thunk) in nodeCalcs:
                try:
                    res = thunk()
                    if res:
                        if not isinstance(res, str):
                            print 'Not a string:', name
                            res = str(res)
                        nodes.append(template({name: res}))
                except:
                    pass

            frag = combine_nodes_into_fragment(nodes)

            # Wrap the fragment and return it.
            return hst_parameters({'hst_parameters': frag})
