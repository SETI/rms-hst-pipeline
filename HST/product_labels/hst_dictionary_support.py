##########################################################################################
# hst_dictionary_support.py
##########################################################################################

import os

import astropy.io.fits as pyfits
import pdslogger

WFPC2_DETECTOR_IDS = {1: 'PC1', 2: 'WF2', 3: 'WF3', 4: 'WF4'}
WFPC_DETECTOR_IDS  = {1: 'WF1', 2: 'WF2', 3: 'WF3', 4: 'WF4',
                      5: 'PC5', 6: 'PC6', 7: 'PC7', 8: 'PC8'}

WFPC2_TARGETED_DETECTOR_IDS = {
    'F160BN15'  : ['WF3'],
    'FQCH4N1'   : ['PC1', 'WF3'],
    'FQCH4N15'  : ['PC1'],
    'FQCH4N33'  : ['WF2', 'WF3'],
    'FQCH4W2'   : ['WF2'],
    'FQCH4W3'   : ['WF3'],
    'FQCH4W4'   : ['WF4'],
    'FQCH4P15'  : ['PC1'],
    'FQUVN33'   : ['WF2'],
    'PC1'       : ['PC1'],
    'POLQN18'   : ['WF2'],
    'POLQN33'   : ['WF2'],
    'POLQP15P'  : ['PC1'],
    'POLQP15W'  : ['WF2'],
    'WF2'       : ['WF2'],
    'WF3'       : ['WF3'],
    'WF4'       : ['WF4'],
    'WFALL'     : ['PC1', 'WF2', 'WF3', 'WF4'],
}

PLATE_SCALES = {  # plate scales in arcsec/pixel
    ('ACS', 'HRC'): 0.026,
    ('ACS', 'SBC'): 0.032,
    ('ACS', 'WFC1'): 0.05,
    ('ACS', 'WFC2'): 0.05,
    ('NICMOS', 'NIC1'): 0.042,
    ('NICMOS', 'NIC2'): 0.075,
    ('NICMOS', 'NIC3'): 0.2,
    ('WFPC', 'WF'): 0.1016,
    ('WFPC', 'PC'): 0.0439,
    ('WFPC2', 'PC1'): 0.046,
    ('WFPC2', 'WF2'): 0.1,
    ('WFPC2', 'WF3'): 0.1,
    ('WFPC2', 'WF4'): 0.1,
}

FOC_OBSERVATION_TYPE_FROM_OPMODE = {
    'ACQ'  : 'IMAGING',
    'IMAGE': 'IMAGING',
    'OCC'  : 'TIME-SERIES',
    'SPEC' : 'SPECTROSCOPIC',
}

FOC_PLATE_SCALES = {    # (instrument_mode_id, observation_type) from Table 7a,b,c
    ('F96/ZOOM', 'IMAGING'       ): 0.014,
    ('F96'     , 'IMAGING'       ): 0.014,
    ('F96/ZOOM', 'TIME-SERIES'   ): 0.014,
    ('F96'     , 'TIME-SERIES'   ): 0.014,
    ('F48/ZOOM', 'IMAGING'       ): 0.028,
    ('F48'     , 'IMAGING'       ): 0.028,
    ('F48/ZOOM', 'SPECTROSCOPIC' ): 0.056,
    ('F48'     , 'SPECTROSCOPIC' ): 0.028,
}

FOS_APERTURE_NAMES = {
    'A-1': '4.3 (A-1)',
    'A-2': '0.5-PAIR (A-2)',
    'A-3': '0.25-PAIR (A-3)',
    'A-4': '0.1-PAIR (A-4)',
    'B-1': '0.5 (B-1)',
    'B-2': '0.3 (B-2)',
    'B-3': '1.0 (B-3)',
    'B-4': 'BLANK (B-4)',
    'C-1': '1.0-PAIR (C-1)',
    'C-2': '0.25X2.0 (C-2)',
    'C-3': '2.0-BAR (C-3)',
    'C-4': '0.7X2.0-BAR (C-4)',
    'FAILSAFE': 'FAILSAFE',
}

FOS_FILTER_NAME_FROM_FGWA_ID = {
    'H13': 'G130H',
    'L15': 'G160L',
    'H19': 'G190H',
    'H27': 'G270H',
    'H40': 'G400H',
    'H57': 'G570H',
    'L65': 'G650L',
    'H78': 'G780H',
    'PRI': 'PRISM',
    'CAM': 'MIRROR',
}

GHRS_WAVELENGTHS = {        # in Angstroms, from Table 34.1
    'G140L'     : ((1100+1900)/2., (0.572+0.573)/2.),
    'G140M'     : ((1100+1900)/2., (0.056+0.052)/2.),
    'G160M'     : ((1150+2300)/2., (0.072+0.066)/2.),
    'G200M'     : ((1600+2300)/2., (0.081+0.075)/2.),
    'G270M'     : ((2000+3300)/2., (0.096+0.087)/2.),
    'Echelle A' : ((1100+1700)/2., (0.011+0.018)/2.),
    'Echelle B' : ((1700+3200)/2., (0.017+0.034)/2.),
}

INSTRUMENT_NAME = {
    'ACS'   : 'Advanced Camera for Surveys',
    'COS'   : 'Cosmic Origins Spectrograph',
    'FGS'   : 'Fine Guidance Sensors',
    'FOC'   : 'Faint Object Camera',
    'FOS'   : 'Faint Object Spectrograph',
    'GHRS'  : 'Goddard High Resolution Spectrograph',
    'HSP'   : 'High Speed Photometer',
    'NICMOS': 'Near Infrared Camera and Multi-Object Spectrometer',
    'STIS'  : 'Space Telescope Imaging Spectrograph',
    'WFC3'  : 'Wide Field Camera 3',
    'WFPC'  : 'Wide Field and Planetary Camera',
    'WFPC2' : 'Wide Field and Planetary Camera 2',
}

def _join_hst_pi_name(last, first, middle):
    """Merge the last name, first name, and middle initial as extracted from an HST FITS
    header and attempt to correct the capitalization.
    """

    def capitalize_name(name):
        if name == name.upper():
            name = name.capitalize()

            for prefix in ('Mc', 'Mac'):
                if name.startswith(prefix):
                    name = prefix + name[len(prefix):].capitalize()

            chars = list(name)
            for i,c in enumerate(chars[:-1]):
                if not c.isalpha():
                    chars[i+1] = chars[i+1].upper()
            name = ''.join(chars)

        return name

    last = capitalize_name(last)
    first = capitalize_name(first)

    if len(middle) == 1:
        middle += '.'

    return f'{last}, {first} {middle}'.rstrip()

def fill_hst_dictionary(ref_hdulist, spt_hdulist, filepath='', logger=None):
    """Return a dictionary containing all of the values in the HST dictionary.

    Input:
        ref_hdulist     HDU list for the reference data file, as returned by
                        astropy.io.fits.open(). If there is no such file, use None.
        spt_hdulist     HDU for the spt/shm/shf file associated with this data file.
        filepath        name of the reference file, primarily for error logging.
        logger          pdslogger to use.

    Note: For COS/FUV, we need to check the file system to determine whether the suffixes
    "_a" and "_b" both exist; this is the only reliable mechanism to determine the full
    list of detector_ids. For this reason, the given filepath needs to be a full path
    to one of the data files.

    For ACS/WFC, WFC3/UVIS, and WFPC2, the returned dictionary contains an additional item
    keyed by "detector_id_vs_extver", which is dictionary that returns the detector_id
    associated with each EXTVER in a label.
    """

    def get_or_log(header, key, alt='UNK'):
        """Internal function to look up in given FITS header; log error on failure.
        """
        try:
            return header[key]
        except KeyError:
            logger.error('Missing FITS keyword ' + key, filepath)
            return alt

    logger = logger or pdslogger.NullLogger()

    # Ignore missing ref_hdulist if SCIDATA is False---just for GHRS
    spt_header = spt_hdulist[0].header
    if ref_hdulist is not None:
        scidata = True
        header0 = ref_hdulist[0].header
        if len(ref_hdulist) > 1:
            header1 = ref_hdulist[1].header
        else:
            header1 = {}
    else:
        scidata = False
        header0 = {}
        header1 = {}

    # Create an input dictionary with merged content of the above three headers
    merged = {}
    for (key, value) in spt_header.items():
        merged[key] = value
    for (key, value) in header1.items():
        merged[key] = value
    for (key, value) in header0.items():
        merged[key] = value

    # Create a dictionary with "UNK" content
    hst_dictionary = {
        'aperture_name'                 : 'UNK',
        'bandwidth'                     : 0.,
        'binning_mode'                  : 'UNK',
        'center_filter_wavelength'      : 0.,
        'channel_id'                    : 'UNK',
        'coronagraph_flag'              : 'UNK',
        'cosmic_ray_split_count'        : 0,
        'detector_ids'                  : ['UNK'],
        'exposure_duration'             : 0.,
        'exposure_type'                 : 'UNK',
        'filter_name'                   : 'UNK',
        'fine_guidance_sensor_lock_type': 'UNK',
        'gain_setting'                  : 0.,
        'gyroscope_mode'                : 0,
        'hst_pi_name'                   : 'UNK',
        'hst_proposal_id'               : 0,
        'hst_quality_comment'           : 'UNK',
        'hst_quality_id'                : 'UNK',
        'hst_target_name'               : 'UNK',
        'instrument_id'                 : 'UNK',
        'instrument_mode_id'            : 'UNK',
        'mast_observation_id'           : 'UNK',
        'mast_pipeline_version_id'      : 'UNK',
        'moving_target_descriptions'    : ['UNK'],
        'moving_target_flag'            : 'UNK',
        'moving_target_keywords'        : ['UNK'],
        'observation_type'              : 'UNK',
        'plate_scale'                   : 0.,
        'proposed_aperture_name'        : 'UNK',
        'repeat_exposure_count'         : 0,
        'spectral_resolution'           : 0.,
        'subarray_flag'                 : 'UNK',
        'targeted_detector_ids'         : ['UNK'],
        'visit_id'                      : 'UNK',
    }

    ##############################
    # instrument_id
    ##############################

    instrument_id = merged.get('INSTRUME', '')
    if instrument_id == 'HRS':
        instrument_id = 'GHRS'

    hst_dictionary['instrument_id'] = instrument_id

    ##############################
    # channel_id
    # ACS: HRC, SBN, WFC
    # COS: FUV, NUV
    # FOS: AMBER, BLUE
    # GHRS: D1, D2
    # HSP: always HSP
    # NICMOS: NIC1, NIC2, NIC3
    # WFPC: WF, PC
    # WFC3: UVIS, IR
    ##############################

    try:
        if instrument_id == 'NICMOS':
            channel_id = 'NIC' + str(header0['CAMERA'])
        elif instrument_id == 'WFPC':
            channel_id = header0['CAMERA']
        elif instrument_id == 'HSP':
            channel_id = 'HSP'
        elif instrument_id == 'GHRS':
            channel_id = 'D' + str(spt_header['SS_DET'])
        else:
            try:
                ccd = header0['DETECTOR']
                if instrument_id == 'WFPC2':
                    channel_id = WFPC2_DETECTOR_IDS[ccd]
                else:
                    channel_id = ccd
            except KeyError:
                channel_id = instrument_id

        hst_dictionary['channel_id'] = channel_id

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # detector_ids
    # ACS/WFC: WFC1, WFC2
    # COS/FUV: FUVA, FUVB
    # FGS: FGS1, FGS2, FGS3, FGS1R after 1997-02-11; FGS2R after 1999-12-19
    # HSP: VIS, POL, UV1, UV2, PMT
    # WFPC: PC1, PC2, PC3, PC4, WF1, W2, WF3, WF4
    # WFC3/UVIS: UVIS1, UVIS2
    # otherwise, the channel ID
    ##############################

    # For ACS/WFC, WFC3/UVIS, and WFPC2, the returned dictionary contains an additional
    # item "detector_id_vs_extver", which is dictionary that returns the detector_id
    # associated with each EXTVER in a label.

    # For WFPC, the returned dictionary contains an additional item
    # "detector_id_vs_axis", which is a list that indicates the detector_id for each
    # "layer" in the 3-D image array.

    # Interior function
    def detector_number_vs_extver(hdulist, fitsname):
        detector_number_dict = {}
        max_extver = 0
        for hdu in hdulist:
            extver = hdu.header.get('EXTVER', 0)
            max_extver = max(extver, max_extver)

            detector_number = hdu.header.get(fitsname, 0)
            try:
                detector_number = int(detector_number)
            except ValueError:
                detector_number = 0

            # Ignore zero-valued detector numbers
            if detector_number:
                detector_number_dict[extver] = detector_number

        return detector_number_dict

    detector_ids = []
    try:
        if (instrument_id, channel_id) in {('ACS', 'WFC'), ('WFC3', 'UVIS')}:
            detector_number_dict = detector_number_vs_extver(ref_hdulist, 'CCDCHIP')
            if -999 in detector_number_dict.values():
                detector_numbers = [1, 2]
            else:
                detector_numbers = list(set(detector_number_dict.values()))
                detector_numbers.sort()

            detector_ids = [f'{channel_id}{n}' for n in detector_numbers if n in {1,2}]

            detector_id_vs_extver = {extver:(f'{channel_id}{n}' if n in {1,2} else '')
                                     for extver,n in detector_number_dict.items()}
            hst_dictionary['detector_id_vs_extver'] = detector_id_vs_extver

        elif instrument_id == 'WFPC2':
            detector_number_dict = detector_number_vs_extver(ref_hdulist, 'DETECTOR')
            detector_numbers = list(set(detector_number_dict.values()))
            detector_numbers.sort()
            detector_ids = [WFPC2_DETECTOR_IDS[n] for n in detector_numbers
                            if n in {1,2,3,4}]

            detector_id_vs_extver = {key:WFPC2_DETECTOR_IDS.get(n, '')
                                     for key,n in detector_number_dict.items()}
            hst_dictionary['detector_id_vs_extver'] = detector_id_vs_extver

        elif instrument_id == 'WFPC':
            # The reference file uses a waivered format, with a 3-D image in which there
            # is one image layer for each detector used. The second HDU contains a table
            # indicating the detector associated with each layer.
            detector_numbers = ref_hdulist[1].data['DETECTOR']
            detector_ids = [WFPC_DETECTOR_IDS[n] for n in detector_numbers]

        elif instrument_id == 'COS' and channel_id == 'FUV':
            # We need to check the file system for both "_a" and "_b" suffixes
            if not os.path.exists(filepath):
                logger.error('Invalid path to COS/FUV file', filepath)
                detector_ids = ['UNK']
            elif filepath.endswith('_a.fits'):
                if os.path.exists(filepath.replace('_a.fits', '_b.fits')):
                    detector_ids = ['FUVA', 'FUVB']
                else:
                    detector_ids = ['FUVA']
                    logger.info('Note: COS/FUVB file not found', filepath)
            elif filepath.endswith('_b.fits'):
                if os.path.exists(filepath.replace('_b.fits', '_a.fits')):
                    detector_ids = ['FUVA', 'FUVB']
                else:
                    detector_ids = ['FUVB']
                    logger.info('Note: COS/FUVA file not found', filepath)

            # Otherwise, see if the second HDU is a table and "SEGMENT" is a column
            elif isinstance(ref_hdulist[1].data, pyfits.fitsrec.FITS_rec):
                try:
                    detector_ids = list(ref_hdulist[1].data['SEGMENT'])
                except KeyError:
                    logger.error('COS/FUV table does not have a column "SEGMENT"',
                                 filepath)
                    detector_ids = ['UNK']

            else:
                logger.error('COS/FUV file does not contain "_a" or "_b"', filepath)
                detector_ids = ['UNK']

        elif instrument_id == 'FGS':
            date = merged.get('DATE-OBS', '9999')
            if date < '1997-02-11':
                detector_ids = ['FGS1', 'FGS2', 'FGS3']
            elif date < '1999-12-19':
                detector_ids = ['FGS1R', 'FGS2', 'FGS3']
            else:
                detector_ids = ['FGS1R', 'FGS2R', 'FGS3']

        elif instrument_id == 'HSP':
            config = spt_header['CONFIG']
            # Example: config = HSP/UNK/VIS
            parts = config.split('/')
            if parts[0] != 'HSP':
                logger.error('Invalid HSP CONFIG value ' + config, filepath)
                detector_ids = ['UNK']
            else:
                detector_ids = [p for p in parts[1:] if p != 'UNK']

        # Otherwise, return the single value of channel_id
        else:
            detector_ids = [channel_id]

        hst_dictionary['detector_ids'] = detector_ids

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # aperture_name
    ##############################

    try:
        if instrument_id in ('WFPC', 'WFPC2', 'HSP', 'GHRS'):
            aperture_name = spt_header['APER_1']
        elif instrument_id == 'FGS':
            aperture_name = 'FGS' + header0['FGSID']
        elif instrument_id == 'FOC':
            aperture_name = ''      # not applicable for FOC
        elif instrument_id == 'FOS':
            aperture_name = FOS_APERTURE_NAMES[header0['APER_ID']]
        else:
            # This is valid for most instruments
            # aperture_name = get_or_log(header0, 'APERTURE')
            # aperture_name is nillable, if 'APERTURE' doesn't exist, set to 'UNK'
            # nilReason="missing"
            aperture_name = header0.get('APERTURE', 'UNK')

        hst_dictionary['aperture_name'] = aperture_name

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # bandwidth
    ##############################

    # Works for STIS and WFPC2; otherwise, 0.
    try:
        hst_dictionary['bandwidth'] = float(merged.get('BANDWID', 0.)) * 1.0e-4

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # binning_mode
    ##############################

    binning_mode = 1    # default

    try:
        # WFPC and WFPC2 are special cases
        if instrument_id in ('WFPC', 'WFPC2'):
            obsmode = header0['MODE']
            if obsmode == 'AREA':
                binning_mode = 2
        else:
            # Binning info can be in the first or second FITS header
            try:
                binaxis1 = merged['BINAXIS1']
                binaxis2 = merged['BINAXIS2']
                binning_mode = max(binaxis1, binaxis2)
            except KeyError:
                pass

        hst_dictionary['binning_mode'] = binning_mode

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # coronagraph_flag
    ##############################

    if instrument_id == 'ACS':
        coronagraph_flag = (aperture_name.startswith('HRC-CORON') or
                            aperture_name.startswith('HRC-OCCULT'))
    elif instrument_id == 'STIS':
        coronagraph_flag = (aperture_name == '50CORON'
                            or aperture_name.startswith('BAR')
                            or aperture_name.startswith('WEDGE')
                            or aperture_name.startswith('52X0.2F1'))
    elif instrument_id == 'NICMOS':
        coronagraph_flag = (aperture_name == 'NIC2-CORON')

    else:
        coronagraph_flag = False

    hst_dictionary['coronagraph_flag'] = coronagraph_flag

    ##############################
    # cosmic_ray_split_count
    ##############################

    try:
        cosmic_ray_split_count = header0['CRSPLIT']
    except KeyError:
        cosmic_ray_split_count = 1  # no CR-splitting unless explicitly stated

    hst_dictionary['cosmic_ray_split_count'] = cosmic_ray_split_count

    ##############################
    # exposure_duration
    ##############################

    if scidata:
        exposure_duration = get_or_log(merged, 'EXPTIME', 0.)
        exposure_duration = round(exposure_duration, 3)     # strip extraneous precision
        hst_dictionary['exposure_duration'] = exposure_duration

    ##############################
    # exposure_type
    ##############################

    if scidata:
        hst_dictionary['exposure_type'] = get_or_log(merged, 'EXPFLAG', 0.)

    ##############################
    # filter_name
    ##############################

    try:
        if instrument_id == 'ACS':
            filter1 = header0['FILTER1']
            filter2 = header0['FILTER2']
            if filter1.startswith('CLEAR'):
                if filter2.startswith('CLEAR') or filter2 == 'N/A':
                    filter_name = 'CLEAR'
                else:
                    filter_name = filter2
            elif filter2.startswith('CLEAR') or filter2 == 'N/A':
                filter_name = filter1
            else:
                # At this point, both filters start with "F" followed by three digits, or
                # "POL" for polarizers. Sort by increasing wavelength; put polarizers
                # second; join with a plus.
                filters = [filter1, filter2]
                filters.sort()
                filter_name = '+'.join(filters)

        elif instrument_id == 'FGS':
            filter_name = header0['CAST_FLT']

        elif instrument_id == 'FOC':
            filters = [header0['FILTNAM1'], header0['FILTNAM2'],
                       header0['FILTNAM3'], header0['FILTNAM4']]
            filters = [f for f in filters if f]     # ignore blanks
            filters = [f for f in filters if not f.startswith('CLEAR')]
            filters = [f for f in filters if not f.endswith('ND')]
            filters.sort()
            filter_name = '+'.join(filters)

        elif instrument_id == 'FOS':
            filter_name = FOS_FILTER_NAME_FROM_FGWA_ID[header0['FGWA_ID']]
            polar_id = header0['POLAR_ID']
            if polar_id != 'C':
                polar_id += '+' + polar_id

        elif instrument_id == 'HSP':
            filter_name = spt_header['SPEC_1']

        elif instrument_id == 'GHRS':
            if scidata:
                filter_name = header0['GRATING']
            else:
                filter_name = spt_header['SPEC_1']

        elif instrument_id == 'STIS':
            opt_elem = header0['OPT_ELEM']
            filter = header0['FILTER'].upper().replace(' ', '_')
            if filter == 'CLEAR':
                filter_name = opt_elem
            else:
                filter_name = opt_elem + '+' + filter

        elif instrument_id in ('WFPC', 'WFPC2'):
            filtnam1 = header0['FILTNAM1']
            filtnam2 = header0['FILTNAM2']
            if filtnam1 == '':
                filter_name = filtnam2
            elif filtnam2 == '':
                filter_name = filtnam1
            else:
                # At this point, both filters start with "F", followed by three digits.
                # Put lower value first; join with a plus.
                filters = [filtnam1, filtnam2]
                filters.sort()
                filter_name = '+'.join(filters)

        else:
            # For other instruments there is just zero or one filter
            try:
                filter_name = header0['FILTER']
            except KeyError:
                filter_name = ''    # not applicable

        hst_dictionary['filter_name'] = filter_name

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # center_filter_wavelength
    ##############################

    if instrument_id == 'GHRS':
        center_filter_wavelength = GHRS_WAVELENGTHS.get(filter_name, (0.,0.))[0] * 1.e-4

    # Works for STIS and WFPC2; otherwise 0.
    else:
        try:
            center_filter_wavelength = float(header0.get('CENTRWV', 0.)) * 1.e-4
        except Exception:
            center_filter_wavelength = 0.

    hst_dictionary['center_filter_wavelength'] = center_filter_wavelength

    ##############################
    # fine_guidance_sensor_lock_type
    ##############################

    if scidata:
        hst_dictionary['fine_guidance_sensor_lock_type'] = get_or_log(merged, 'FGSLOCK')

    ##############################
    # gain_setting
    ##############################

    try:
        # Works for WFPC2
        if 'ATODGAIN' in header0:
            gain_setting = int(header0['ATODGAIN'] + 0.5)   # 7 or 15

        # Works for ACS, WFC3, others
        elif 'CCDGAIN' in header0:
            gain_setting = float('%3.1f' % float(header0['CCDGAIN']))

        else:
            gain_setting = 0.

        hst_dictionary['gain_setting'] = gain_setting

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # gyroscope_mode
    ##############################

    try:
        gyromode = str(header0.get('GYROMODE', '3')).replace('T', '3')
        hst_dictionary['gyroscope_mode'] = int(gyromode)
    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # hst_pi_name
    ##############################

    try:
        pr_inv_l = merged['PR_INV_L']
        pr_inv_f = merged['PR_INV_F']
        pr_inv_m = merged.get('PR_INV_M', '')
    except KeyError:
        logger.error('Missing FITS keywords PR_INV_*', filepath)
        hst_pi_name = 'UNK'
    else:
        hst_pi_name = _join_hst_pi_name(pr_inv_l, pr_inv_f, pr_inv_m)

    hst_dictionary['hst_pi_name'] = hst_pi_name

    ##############################
    # hst_proposal_id
    ##############################

    hst_dictionary['hst_proposal_id'] = get_or_log(merged, 'PROPOSID')

    ##############################
    # hst_quality_comment
    ##############################

    try:
        comment = (header0['QUALCOM1'] + ' ' +
                   header0['QUALCOM2'] + ' ' +
                   header0['QUALCOM3'])
    except KeyError:
        comment = 'UNK'

    comment = ' '.join(comment.strip().split())
    hst_dictionary['hst_quality_comment'] = comment.strip()

    ##############################
    # hst_quality_id
    ##############################

    hst_dictionary['hst_quality_id'] = header0.get('QUALITY', 'UNK')

    ##############################
    # hst_target_name
    ##############################

    hst_dictionary['hst_target_name'] = get_or_log(merged, 'TARGNAME')

    ##############################
    # instrument_mode_id
    ##############################

    try:
        if instrument_id in ('WFPC', 'WFPC2'):
            instrument_mode_id = header0['MODE']

        elif instrument_id == 'FGS':
            instrument_mode_id = ''         # not applicable for FGS

        elif instrument_id == 'FOC':
            instrument_mode_id = header0['OPTCRLY']
            if header0.get('PXFORMT', '') == 'ZOOM':
                instrument_mode_id + '/ZOOM'

        elif instrument_id == 'FOS':
            instrument_mode_id = header0['GRNDMODE']

        elif instrument_id in ('HSP', 'GHRS'):
            instrument_mode_id = spt_header['OPMODE']

        else:
            # For most HST instruments, this should work...
            # instrument_mode_id = get_or_log(header0, 'OBSMODE')
            # instrument_mode_id is nillable, if 'OBSMODE' doesn't exist, set to 'UNK'
            # nilReason="missing"
            instrument_mode_id = header0.get('OBSMODE', 'UNK')

        hst_dictionary['instrument_mode_id'] = instrument_mode_id

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # mast_observation_id
    ##############################

    mast_observation_id = merged.get('ROOTNAME', '') or merged.get('ASN_ID', '')
    if not mast_observation_id:
        logger.error('Missing FITS keywords ROOTNAME/ASN_ID', filepath)

    hst_dictionary['mast_observation_id'] = mast_observation_id.lower()

    ##############################
    # mast_pipeline_version_id
    ##############################

    hst_dictionary['mast_pipeline_version_id'] = merged.get('OPUS_VER', 'UNK')

    ##############################
    # moving_target_descriptions
    ##############################

    # Defined by 'MT_LVm_n' keyword values in the _spt.fits files
    descs = []
    for k in range(0, 4):
        prefix = 'MT_LV' + ('' if k == '' else str(k)) + '_'
        if not (prefix + '1') in spt_header:
            continue
        desc = []
        for j in range(1, 10):
            keyword = prefix + str(j)
            if keyword not in spt_header:
                break
            desc.append(spt_header[keyword])
        descs.append(''.join(desc))

    hst_dictionary['moving_target_descriptions'] = descs or []

    ##############################
    # moving_target_keywords
    ##############################

    keywords = []
    for k in range(1, 10):
        keyword = 'TARKEY' + str(k)
        try:
            value = spt_header[keyword]
            keywords.append(value)
        except KeyError:
            break

    hst_dictionary['moving_target_keyword'] = keywords or []

    ##############################
    # moving_target_flag
    ##############################

    # Usually in the first FITS header, but in the shf header for GHRS; spt for COS
    try:
        mtflag = merged['MTFLAG']
    except KeyError:
        logger.warn('Missing FITS keyword MTFLAG', filepath)
        mtflag = False

    if mtflag in ('T', '1', True):
        moving_target_flag = True
    elif mtflag in ('F', '', '0', False):
        moving_target_flag = False
    else:
        logger.error(f'Unrecognized MTFLAG value {mtflag}', filepath)
        moving_target_flag = 'UNK'

    hst_dictionary['moving_target_flag'] = moving_target_flag

    ##############################
    # observation_type
    ##############################

    obstype = merged.get('OBSTYPE', '')
    observation_type = obstype
    if observation_type == 'SPECTROGRAPHIC':
        observation_type = 'SPECTROSCOPIC'

    if observation_type not in ('IMAGING', 'SPECTROSCOPIC'):
        if instrument_id in ('ACS', 'FGS', 'NICMOS', 'WFC3', 'WFPC', 'WFPC2'):
            observation_type = 'IMAGING'
        elif instrument_id in ('COS', 'FOS', 'GHRS'):
            observation_type = 'SPECTROSCOPIC'
        elif instrument_id == 'HSP':
            observation_type = 'TIME-SERIES'
        elif instrument_id == 'FOC':
            opmode = spt_header['OPMODE']
            observation_type = FOC_OBSERVATION_TYPE_FROM_OPMODE[opmode]
        elif obstype:
            logger.error('Unrecognized OBSTYPE value ' + obstype, filepath)
            observation_type = 'UNK'
        else:
            logger.error('Missing FITS keyword OBSTYPE', filepath)
            observation_type = 'UNK'

    hst_dictionary['observation_type'] = observation_type

    ##############################
    # proposed_aperture_name
    ##############################

    # Only a few instruments make this distinction
    hst_dictionary['proposed_aperture_name'] = header0.get('PROPAPER', aperture_name)

    ##############################
    # repeat_exposure_count
    ##############################

    hst_dictionary['repeat_exposure_count'] = header0.get('NRPTEXP', 1)

    ##############################
    # plate_scale
    ##############################

    # Works for STIS
    try:
        if 'PLATESC' in header0:
            plate_scale = header0['PLATESC']

        elif instrument_id == 'FOC':
            plate_scale = FOC_PLATE_SCALES[instrument_mode_id, observation_type]
        else:
            # Works for any instrument_id tabulated in the PLATE_SCALEs dictionary
            plate_scales = []
            for detector in detector_ids:
                key = (instrument_id, detector)
                if key in PLATE_SCALES:
                    plate_scales.append(PLATE_SCALES[key] * binning_mode)

            if plate_scales:
                plate_scale = min(plate_scales)
            else:
                plate_scale = 0.

        hst_dictionary['plate_scale'] = plate_scale

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # spectral_resolution
    ##############################

    spectral_resolution = 0.
    if instrument_id == 'GHRS':
        center_filter_wavelength = GHRS_WAVELENGTHS.get(filter_name, (0.,0.))[1] * 1.e-4

    else:
        try:
            if instrument_id == 'STIS':
                spectral_resolution = merged.get('SPECRES', 0.) * 1.0e-4
            elif instrument_id == 'FOC' and spt_header['OPMODE'] == 'SPEC':
                spectral_resolution = 0.00018

        except Exception as e:
            logger.exception(e, filepath)

    hst_dictionary['spectral_resolution'] = spectral_resolution

    ##############################
    # subarray_flag
    ##############################

    flag = str(header0.get('SUBARRAY', '0'))
    if flag == '1' or flag.startswith('T'):
        subarray_flag = True
    elif flag == '0' or flag.startswith('F'):
        subarray_flag = False
    else:
        logger.error('Unrecognized SUBARRAY value ' + flag, filepath)
        subarray_flag = 'UNK'

    hst_dictionary['subarray_flag'] = subarray_flag

    ##############################
    # targeted_detector_ids
    ##############################

    try:
        if instrument_id == 'WFPC2':
            key = aperture_name.replace('-FIX', '')
            try:
                targeted_detector_ids = WFPC2_TARGETED_DETECTOR_IDS[key]
            except KeyError:
                logger.error('Unrecognized WFPC2 aperture ' + aperture_name, filepath)
                targeted_detector_ids = ['UNK']

        elif instrument_id == 'ACS' and channel_id == 'WFC':
            if aperture_name.startswith('WFC1'):
                targeted_detector_ids = ['WFC1']
            elif aperture_name.startswith('WFC2'):
                targeted_detector_ids = ['WFC2']
            else:
                targeted_detector_ids = ['WFC1', 'WFC2']

        elif instrument_id == 'WFC' and channel_id == 'UVIS':
            if aperture_name.startswith('UVIS1'):
                targeted_detector_ids = ['UVIS1']
            elif aperture_name.startswith('UVIS2'):
                targeted_detector_ids = ['UVIS2']
            elif aperture_name.startswith('UVIS-QUAD'):
                filter = header0['FILTER']
                if filter in ('FQ378N', 'FQ387N', 'FQ437N', 'FQ492N', 'FQ508N',
                              'FQ619N', 'FQ674N', 'FQ750N', 'FQ889N', 'FQ937N'):
                    targeted_detector_ids = ['UVIS1']
                elif filter in ('FQ232N', 'FQ243N', 'FQ422M', 'FQ436N', 'FQ575N',
                                'FQ634N', 'FQ672N', 'FQ727N', 'FQ906N', 'FQ924N'):
                    targeted_detector_ids = ['UVIS2']
                else:
                    logger.error('Unrecognized WFC/UVIS quad aperture/filter ' +
                                     f'{aperture_name}/{filter}', filepath)
            else:
                targeted_detector_ids = ['UVIS1', 'UVIS2']

        elif instrument_id == 'WFPC':
            # I cannot find documentation for the apertures for WFPC so this is just an
            # educated guess
            if aperture_name in ('ALL', 'ANY'):
                targeted_detector_ids = detector_ids
            elif aperture_name.startswith('W'):
                targeted_detector_ids = ['WF' + aperture_name[1]]  # WFn
            elif aperture_name.startswith('P'):
                targeted_detector_ids = ['PC' + aperture_name[1]]  # PCn
            else:
                logger.error('Unrecognized WFPC aperture ' + aperture_name, filepath)
                targeted_detector_ids = ['UNK']

        elif instrument_id == 'FGS':
            digit = int(header0['FGSID'])
            targeted_detector_ids = [detector_ids[digit-1]]

        elif instrument_id == 'HSP':    # when HSP uses multiple detectors, the first
                                        # detector listed applies to the star and the
                                        # second to the sky.
            targeted_detector_ids = [detector_ids[0]]

        else:
            targeted_detector_ids = detector_ids

        hst_dictionary['targeted_detector_ids'] = targeted_detector_ids

    except Exception as e:
        logger.exception(e, filepath)

    ##############################
    # visit_id
    ##############################

    basename = os.path.basename(filepath)
    hst_dictionary['visit_id'] = basename[4:6]

    ############################################################
    # Return the dictionary
    ############################################################
    return hst_dictionary

##########################################################################################

def hst_detector_ids_for_file(hst_dictionary, filepath):
    """Return the list of detector IDs for the given file path. For a few older
    instruments with multiple detectors, certain FITS files contain data from only one of
    the detectors used, and this can be inferred from the basename.

    Input:
        hst_dictionary  HST dictionary for this IPPPSSOOT.
        filepath        path to this file.
    """

    instrument_id = hst_dictionary['instrument_id']
    channel_id = hst_dictionary['channel_id']
    detector_ids = hst_dictionary['detector_ids']

    if len(detector_ids) == 1:
        return detector_ids

    if instrument_id == 'COS' and channel_id == 'FUV':
        if filepath.endswith('_a.fits'):
            return ['FUVA']

        if filepath.endswith('_b.fits'):
            return ['FUVB']

    elif instrument_id == 'HSP':
        if filepath[-7:] in ('0f.fits', '2f.fits'):     # star data from first detector
            return [detector_ids[0]]

        if filepath[-7:] in ('1f.fits', '3f.fits'):     # sky data from second detector
            return [detector_ids[-1]]

    elif instrument_id == 'FGS':
        digit = filepath[-7]
        if digit in '123':
            return [detector_ids[int(digit)-1]]

    return detector_ids

##########################################################################################
