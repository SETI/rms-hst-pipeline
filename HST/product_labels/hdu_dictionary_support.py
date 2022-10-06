##########################################################################################
# hdu_dictionary_support.py
##########################################################################################

import os
import re

# Translate column formats to PDS4 types
TFORM_INFO = {      # (bytes if defined, PDS4 Field_Binary type)
    ('A', 'BINTABLE'): ( 0, 'ASCII_String'),
    ('L', 'BINTABLE'): ( 1, 'ASCII_Boolean'),
    ('B', 'BINTABLE'): ( 1, 'UnsignedByte'),
    ('I', 'BINTABLE'): ( 2, 'SignedMSB2'),
    ('J', 'BINTABLE'): ( 4, 'SignedMSB4'),
    ('K', 'BINTABLE'): ( 8, 'SignedMSB8'),
    ('E', 'BINTABLE'): ( 4, 'IEEE754MSBSingle'),
    ('D', 'BINTABLE'): ( 8, 'IEEE754MSBDouble'),
    ('C', 'BINTABLE'): ( 8, 'ComplexMSB8'),
    ('M', 'BINTABLE'): (16, 'ComplexMSB16'),

    ('A', 'TABLE'): (0, 'ASCII_String'),
    ('I', 'TABLE'): (0, 'ASCII_Integer'),
    ('F', 'TABLE'): (0, 'ASCII_Real'),
    ('E', 'TABLE'): (0, 'ASCII_Real'),
    ('D', 'TABLE'): (0, 'ASCII_Real'),
}

# Translate array formats to PDS4 types
BITPIX_INFO = {     # (bytes per element, PDS4 Field_Binary type)
      8: (1, 'UnsignedByte'),
     16: (2, 'SignedMSB2'),
     32: (4, 'SignedMSB4'),
     64: (8, 'SignedMSB8'),
    -32: (4, 'IEEE754MSBSingle'),
    -64: (8, 'IEEE754MSBDouble'),
}

SCIENCE_ARRAY_CLASS_SUFFIX = {
    'ACS'   : '_Image',
    'COS'   : '_Spectrum',
    'FGS'   : '',
    'FOC'   : '_Image',
    'FOS'   : '_Spectrum',
    'GHRS'  : '_Spectrum',
    'HSP'   : '',
    'NICMOS': '_Image',
    'STIS'  : '_Spectrum',
    'WFC3'  : '_Image',
    'WFPC'  : '_Image',
    'WFPC2' : '_Image',
}

# Used to normalize units and to recognized them when embedded inside column descriptions
UNIT_TRANSLATOR = {
    ''                      : '',
    'angstrom'              : 'Angstrom',
    'Angstroms'             : 'Angstrom',
    'ANGSTROMS'             : 'Angstrom',
    'arcsec'                : 'arcsec',
    'count'                 : 'count',
    'counts'                : 'count',
    'Count'                 : 'count',
    'count/s'               : 'count/s',
    'count/s/pixel'         : 'count/s/pixel',
    'COUNTS'                : 'count',
    'Counts/s'              : 'count/s',
    'COUNTS/S'              : 'count/s',
    'CT/S'                  : 'count/s',
    'CT/S/ARCSEC**2'        : 'count/s/arcsec**2',
    'deg'                   : 'deg',
    'degC'                  : 'degC',
    'degree'                : 'deg',
    'degrees'               : 'deg',
    'DN'                    : 'DN',
    'DN/s'                  : 'DN/s',
    'DN/sec'                : 'DN/s',
    'erg/s/cm**2/angstrom'  : 'erg/s/cm**2/Angstrom',
    'erg/s/cm**2/Angstrom'  : 'erg/s/cm**2/Angstrom',
    'erg/s/cm**2/angstrom/arcsec**2': 'erg/s/cm**2/Angstrom/arcsec**2',
    'electron'              : 'electron',
    'ELECTRON'              : 'electron',
    'electrons'             : 'electron',
    'ELECTRONS'             : 'electron',
    'electron/s'            : 'electron/s',
    'ELECTRON/S'            : 'electron/s',
    'electrons/s'           : 'electron/s',
    'ELECTRONS/S'           : 'electron/s',
    'ERGS/CM**2/S/A'        : 'erg/s/cm**2/Angstrom',
    'Gauss'                 : 'Gauss',
    'km/s'                  : 'km/s',
    'milliarcsec'           : 'milliarcsec',
    'pixel'                 : 'pixel',
    'pixels'                : 'pixel',
    'sample'                : 'sample',
    'SAMPLES'               : 'sample',
    's'                     : 's',
    'seconds'               : 's',
    'SECONDS'               : 's',
    'UNITLESS'              : '',
    'V mag/arcsec2'         : 'Vmag/arcsec**2',
    'V'                     : 'V',
    'Vmag/arcsec**2'        : 'Vmag/arcsec**2',
    'Vmag/arcsec2'          : 'Vmag/arcsec**2',
    'Volts'                 : 'V',
}

# Don't issue warnings when these appear at the end of a description
NOT_UNITS = {"0", "1", "1 or 0", "T/F"}

# These will be called Array_*D_Image or Array_*D_Spectrum rather than Array_*D
SCIENCE_EXTNAMES = {'SCI', 'ERR', 'DQ', 'SAMP', 'TIME', 'WHT', 'CTX'}

def fill_hdu_dictionary(hdu, index, instrument_id, filepath, logger=None):
    """Return a dictionary containing the file structure info needed to describe one
    header object and one data object in an HST FITS file.

    Input:
        hdu             one FITS Header Data Unit as returned by astropy.io.fits.open().
        index           index of this HDU in the original HDU list returned by open().
        instrument_id   instrument ID.
        filepath        directory path to this file.
        logger          pdslogger to use; None to suppress logging.

    Return:             hdu_dict
        hdu_dict        a dictionary containing all the structural info needed to describe
                        this header and this data object in the PDS4 XML label.

        hdu_dict["extname"]
                        EXTNAME of this HDU, possibly edited, e.g., "SCI" or "TRL".

        hdu_dict["extver"]
                        EXTVER of this HDU, or zero if absent.

        hdu_dict["name"]
                        name of this HDU based on EXTNAME and EXTVER, e.g., "SCI_1".
                        Always "PRIMARY" for the first HDU. Blank if no EXTNAME was
                        provided.

        hdu_dict["index"]
                        index of this HDU, starting with zero.

        hdu_dict["header"]
                        a dictionary keyed by the needed attributes of the Header object:
                            "name", "local_identifier", "offset", "object_length",
                            "parsing_standard_id", "description".

        hdu_dict["data"]
                        a dictionary keyed by the needed attributes of the data object,
                        always including:
                            "name", "local_identifier", "offset", "description",
                            "data_class", "is_empty".

        hdu_dict["data"]["data_class"]
                        the class of the data object, one of "Array_1D", "Array_2D",
                        "Array_2D_Image", "Table_Binary", or "" if no data object is
                        present.

        If hdu_dict["data"]["data_class"] begins with "Array", these keys are also
                        included in hdu_dict["data"]:
                            "axes", "axis_index_order", "data_type", "unit",
                            "scaling_factor", "value_offset", "pixvalue".
                        Also, these are lists with one value for each axis of the array:
                            "axis_names", "elements", "sequence_numbers".

        If hdu_dict["data"]["data_class"] is "Table_Binary", these keys are also included
                        in hdu_dict["data"]:
                            "records", "fields", "groups" (always 0), "record_length".
                        Also, these are lists with one value for each column in the table:
                            "names", "field_numbers", "field_locations", "data_types",
                            "field_lengths", "field_formats", "units", "descriptions".

        If hdu_dict["data"]["data_class"] is "", this key is also included:
                            "pixvalue".
                        This is needed because sometimes a missing data object is really
                        a pseudo-array of constant value given by FITS header keyword
                        "PIXVALUE". hdu_dict["data"]["pixvalue"] contains that value if it
                        was provided; otherwise, its value is None.
    """

    logger = logger or pdslogger.NullLogger()

    header = hdu.header
    data = hdu.data

    fileinfo = hdu.fileinfo()
    header_offset = fileinfo['hdrLoc']
    data_offset = fileinfo['datLoc']

    # Extract the suffix from the file basename
    basename = os.path.basename(filepath)
    suffix = basename.partition('.')[0]     # before '.fits'
    suffix = suffix.partition('_')[2]       # after the underscore

    #### Construct the names and descriptions

    extname = header.get('EXTNAME', '')
    extver = header.get('EXTVER', 0)
    if extname:

        # Deal with weird EXTNAMEs in CGR, DGR, SHF, TRL files
        # "n4wl02ltq.trl" -> TRL
        # "v0xt0101t.shh.tab" -> SHH
        # "X3GFC101T_CVT.SHH.TAB" -> SHH
        # "f5br0101m_cvt.a1h.tab" -> A1H
        if '.' in extname:
            extname = extname.partition('.')[2].upper()
            if extname.endswith('.TAB'):
                extname = extname[:-4]
        else:
            extname = extname.upper()

        name = extname + ('_' + str(extver) if extver else '')
        dname = f'FITS data object #{index} ("{name}")'
        hname = f'FITS header #{index} ("{name}")'
        hdesc = hname + '.'

    elif index == 0:
        name = 'PRIMARY'
        dname = 'Primary FITS data object'
        hname = 'PRIMARY FITS header'
        hdesc = 'Primary FITS header.'

    else:
        name = ''
        dname = f'FITS data object #{index}'
        hname = f'FITS header #{index}'
        hdesc = hname + '.'

    #### Initialize the dictionaries

    header_dict = {     # header dictionary
        'name'               : hname,
        'local_identifier'   : f'fits_header_{index}',
        'offset'             : header_offset,
        'object_length'      : data_offset - header_offset,
        'parsing_standard_id': 'FITS 3.0',
        'description'        : hdesc,
    }

    data_dict = {       # data dictionary
        'name'            : dname,
        'local_identifier': f'fits_data_object_{index}',
        'offset'          : data_offset,
        'description'     : dname + '.',
        'is_empty'        : False,
        'pixvalue'        : None,
    }

    info = {            # overall dictionary to be returned
        'name'   : name,
        'extname': extname,
        'extver' : extver,
        'index'  : index,
        'data'   : data_dict,
        'header' : header_dict,
    }

    #### Describe the data class...

    xtension = header.get('XTENSION', '')
    naxis = header.get('NAXIS', 0)

    #### No data object

    if naxis == 0:
        data_dict['data_class'] = ''
        data_dict['is_empty'] = True
        data_dict['pixvalue'] = header.get('PIXVALUE', None)

    #### IMAGE case or waivered file with data object in HDU 0

    elif xtension in ('IMAGE', '', 'HDRLET'):
        naxis1 = header['NAXIS1']
        naxis2 = header.get('NAXIS2', 0)
        naxis3 = header.get('NAXIS3', 0)

        # Merge leading axes above three
        if naxis > 3:
            logger.error(f'NAXIS={naxis} in HDU {index}; merged to NAXIS=3', filepath)
            for k in range(4, naxis+1):
                naxis3 *= header[f'NAXIS{k}']
            naxis = 3

        # Ignore a unit third axis
        if naxis == 3 and naxis3 == 1:
            naxis = 2

        # Convert 2D to 1D if appropriate
        if naxis == 2 and naxis1 == 1 or naxis2 == 1:
            naxis = 1
            naxis1 = max(naxis1, naxis2)

        # Get the data class and axes
        if naxis == 1:
            data_class = 'Array_1D'
            elements = [naxis1]
            axis_names = ['Sample']
            is_empty = (naxis1 == 0)
        elif naxis == 2:
            data_class = 'Array_2D'
            elements = [naxis2, naxis1]
            axis_names = ['Line', 'Sample']
            is_empty = (naxis1 * naxis2 == 0)
        else:
            data_class = 'Array_3D'
            elements = [naxis3, naxis2, naxis1]
            axis_names = ['Detector', 'Line', 'Sample']
            is_empty = (naxis1 * naxis2 * naxis3 == 0)

        if extname in SCIENCE_EXTNAMES or index == 0:
            data_class += SCIENCE_ARRAY_CLASS_SUFFIX[instrument_id]

        # Clean up the units
        unit = header.get('BUNIT', '').replace(' ','')
        if unit in UNIT_TRANSLATOR:
            unit = UNIT_TRANSLATOR[unit]
        else:
            logger.warn(f'Unrecognized array unit "{unit}" in HDU[{index}]',
                        filepath)

        data_dict['data_class'] = data_class
        data_dict['axes'] = naxis
        data_dict['axis_index_order'] = 'Last Index Fastest'
        data_dict['data_type'] = BITPIX_INFO[header['BITPIX']][1]
        data_dict['unit'] = unit
        data_dict['scaling_factor'] = header.get('BSCALE', 1.)
        data_dict['value_offset'] = header.get('BZERO', 0.)
        data_dict['axis_names'] = axis_names
        data_dict['elements'] = elements
        data_dict['sequence_numbers'] = list(range(1,naxis+1))
        data_dict['is_empty'] = is_empty

        if is_empty and 'PIXVALUE' in header:
            data_dict['pixvalue'] = header['PIXVALUE']

    #### TABLE/BINTABLE case

    elif xtension in ('TABLE', 'BINTABLE'):

        data_dict['data_class'] = 'Table_Binary'
        data_dict['records'] = header['NAXIS2']

        # Record_Binary
        columns = header['TFIELDS']
        data_dict['fields'] = columns
        data_dict['groups'] = 0
        data_dict['record_length'] = header['NAXIS1']

        # Field_Binary...
        names = []
        field_numbers = []
        field_locations = []
        data_types = []
        field_lengths = []
        field_formats = []  # Pattern: %[\+,-]?[0-9]+(\.([0-9]+))?[doxfeEs]
        units = []
        descriptions = []

        loc = 1             # track byte location within record
        columns = header['TFIELDS']
        for k in range(1, columns+1):
            suffix = str(k)
            name = header['TTYPE' + suffix]
            tform = header['TFORM' + suffix]
            try:
                (field_length, data_type) = TFORM_INFO[tform[-1], xtension]
                # Note: field_length will be zero here unless it can be inferred from the
                # binary data type
            except KeyError:
                try:
                    (field_length, data_type) = TFORM_INFO[tform[0], xtension]
                except KeyError:
                    logger.error(f'Unsupported TFORM{k}="{tform}" in HDU {index}',
                                 filepath)
                    (field_length, data_type) = (0, 'ASCII_String')

            # Display format field is TFORM for ASCII tables; TDISP for binary tables
            tdisp = header.get('TDISP' + suffix, tform)
            (field_format, alt_length) = pds4_field_format(tdisp)

            # In BINTABLEs, the field length is determined by the type, unless it is a
            # string. For everything else, the field length is defined by the format.
            field_length = field_length or alt_length
            if not field_length:
                logger.error(f'Undefined field length for TFORM{k}="{tform}" ' +
                             f'in HDU {index}', filepath)

            field_location = header.get('TBCOL' + suffix, loc)  # TBCOL is in TABLEs,
                                                                # not BINTABLES
            loc += field_length

            # Older FITS headers provide the definition each column as a FITS keyword
            # based on the column name; newer headers use a comment instead.
            try:
                comments = header.comments['TTYPE' + str(k)]
            except KeyError:
                comments = ''

            if not isinstance(comments, str):  # but the keyword could mean something else
                comments = ''

            # Get description and do initial cleanup
            description = header.get(name, comments)
            if not isinstance(description, str):
                description = ''

            description = ' '.join(description.strip().split())
            if description and isinstance(description, str):
                description = description[0].upper() + description[1:]
            else:
                description = ''

            # Get the unit
            unit = header.get('TUNIT' + suffix, '').replace(' ','').strip()

            # Sometimes the units are appended to the description
            if not unit and description.endswith(')'):
                (short_desc, _, possible_unit) = description[:-1].rpartition('(')
                if possible_unit in UNIT_TRANSLATOR:
                    unit = UNIT_TRANSLATOR[possible_unit]
                    description = short_desc.rstrip()
                elif possible_unit not in NOT_UNITS and '=' not in possible_unit:
                    logger.warn(f'Possible unrecognized unit "({possible_unit})" in ' +
                                f'description for HDU[{index}], column "{name}"',
                                filepath)

            # Sometimes the unit is something weird like "1=TakeData"; move any string
            # like this to the end of the description and clear the units.
            if '=' in unit:
                description += '; ' + unit + '.'
                unit = ''
            elif description and (description[-1] not in ('.?')):
                description += '.'

            # Sometimes there are worthless units like "CHARACTER*8"
            if unit.startswith('CHAR') or unit.startswith('LOGICAL'):
                unit = ''

            # MJD isn't a unit; move to description.
            if unit == 'MJD':
                description = description.rstrip('.') + ' (MJD).'
                unit = ''

            # Standardize the units
            if unit in UNIT_TRANSLATOR:
                unit = UNIT_TRANSLATOR[unit]
            else:
                logger.warn(f'Unrecognized unit "{unit}" for HDU[{index}], ' +
                            f'column "{name}"',
                            filepath)

            names.append(name)
            field_numbers.append(k)
            field_locations.append(header.get('TBCOL' + suffix, loc))
            data_types.append(data_type)
            field_lengths.append(field_length)
            field_formats.append(field_format)
            units.append(unit)
            descriptions.append(description)

        data_dict['names'] = names
        data_dict['field_numbers'] = field_numbers
        data_dict['field_locations'] = field_locations
        data_dict['data_types'] = data_types
        data_dict['field_lengths'] = field_lengths
        data_dict['field_formats'] = field_formats
        data_dict['units'] = units
        data_dict['descriptions'] = descriptions
        data_dict['is_empty'] = (header['NAXIS1'] * header['NAXIS2'] == 0)

    else:
        logger.error(f'Unrecognized XTENSION "{xtension}" in HDU[{index}], ' +
                     f'column "{name}"', filepath)

    # Update the header description if the data object is empty
    if data_dict['is_empty']:
        if data_dict['data_class'] == 'Table_Binary':
            header_dict['description'] += ('\n\nNote that this FITS header describes a ' +
                                           'table with zero rows.')
        elif index == 0:
            header_dict['description'] += (' No data object is associated with this ' +
                                           'FITS header.')
        elif data_dict['pixvalue'] is not None:
            header_dict['description'] += ('\n\nNote that this FITS header describes a ' +
                                           'pseudo-array with a fixed value of ' +
                                           repr(data_dict['pixvalue']) + '.')
        else:
            header_dict['description'] += (' Note that this FITS header describes an ' +
                                           'array of size zero.')

    return info

##########################################################################################

TDISP_REGEX = re.compile(r'([A-Z]+)(\d+)(\.?\d*)(E?\d*)')
WA_REGEX = re.compile(r'(\d+)(A)')

# Tables 15, 16 of the FITS Standard Reference, Version 4.0
# https://fits.gsfc.nasa.gov/fits_standard.html
FIELD_FORMAT_CODES = {
    'A': 's',
    'D': 'E',
    'E': 'E',
    'F': 'f',
    'G': 'E',
    'I': 'd',
    'I': 'd',
    'L': 's',
    'Z': 'x',
    'EN': 'E',
    'ES': 'E',
}

def pds4_field_format(tdisp):
    """PDS4 field_format value from FITS TDISPn or TFORMn value; blank if there's no
    translation.

    Inputs: (Table 20 of FITS standard referenc)
        Aw          character
        Iw[.m]      integer
        Lw          logical 'T' or 'F'
        Ow[.m]      octal
        Zw[.m]      hexadecimal
        Ew.d[Ee]    exponential notation
        ENw.d[Ee]   exponential notation
        ESw.d[Ee]   exponential notation
        Dw.d[Ee]    exponential notation for double precision
    where:
        w = field width
        m = minimum digits
        d = fractional digits
        e = digits in exponent
    Other FITS formats (L,B,G,D) are not supported.

    Alternatively, the input can be of the form "wA", indicating a character string of the
    specified width. "A" alone is treatead as "1A".

    Return:         (format, width)
        format      Allowed PDS4 pattern ("%[\+,-]?[0-9]+(\.([0-9]+))?[doxfeEs]")
        width       character width of the field
    """

    if tdisp == 'A':
        tdisp = '1A'

    match = TDISP_REGEX.fullmatch(tdisp)
    if match:
        (fits_code, digits, frac, expo) = match.groups()
    else:
        match = WA_REGEX.fullmatch(tdisp)
        if not match:
            return ('', 0)
        (digits, fits_code) = match.groups()
        frac = ''
        expo = ''

    try:
        pds_code = FIELD_FORMAT_CODES[fits_code]
    except KeyError:
        return ('', 0)

    return ('%' + digits + frac + pds_code, int(digits))

##########################################################################################

# The EXTVER associated with these EXTNAMES should never be zero
SUPPORT_EXTNAMES = {'ERR', 'DQ', 'SAMP', 'TIME', 'EVENTS', 'GTI', 'TIMELINE', 'WHT',
                    'CTX'}

def repair_hdu_dictionaries(hdu_dicts, filepath, logger):
    """Handle known errors in HST FITS products.
    """

    # Sometimes EXTVER is missing
    # Array_2D_Image is impossible, but it can happen for certain waivered files
    max_extver = 0
    for k, hdu_dict in enumerate(hdu_dicts):
        extname = hdu_dict['extname']
        extver = hdu_dict['extver']
        max_extver = max(max_extver, extver)

        if extname in SUPPORT_EXTNAMES and extver == 0:
            name = f'{extname}_{max_extver}'
            extver = max_extver

            hname = f'FITS header #{k} for data object "{name}"'
            dname = f'{name}: FITS data object #{k}'

            hdu_dict['name'] = name
            hdu_dict['extver'] = extver
            hdu_dict['header']['name'] = hname
            hdu_dict['header']['description'] = hname + '.'
            hdu_dict['data']['name'] = dname
            hdu_dict['data']['description'] = dname + '.'

            logger.warn(f'HDU #{k} name repaired {extname} to {name}', filepath)

        if hdu_dict['data']['data_class'] == 'Array_1D_Image':
            hdu_dict['data']['data_class'] = 'Array_1D'

##########################################################################################
