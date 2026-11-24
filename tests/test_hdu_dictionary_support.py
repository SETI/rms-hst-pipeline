##########################################################################################
# tests/test_hdu_dictionary_support.py
#
# Tests for the data classes that hdu_dictionary_support assigns to FITS data objects.
##########################################################################################

import numpy as np
import pytest
from astropy.io import fits

from hst_pipeline.product_labels.hdu_dictionary_support import (
    fill_hdu_dictionary,
    UNIT_TRANSLATOR,
    science_array_class_suffix,
)


@pytest.mark.parametrize('instrument_id,obstype,expected', [
    ('COS',  'IMAGING',       '_Image'),
    ('COS',  'SPECTROSCOPIC', '_Spectrum'),
    ('COS',  None,            '_Spectrum'),
    ('STIS', 'IMAGING',       '_Image'),
    ('STIS', 'SPECTROSCOPIC', '_Spectrum'),
    ('FOS',  'IMAGING',       '_Spectrum'),
    ('ACS',  'SPECTROSCOPIC', '_Image'),
    ('FGS',  None,            ''),
])
def test_science_array_class_suffix(instrument_id, obstype, expected):
    header = fits.Header()
    if obstype:
        header['OBSTYPE'] = obstype
    assert science_array_class_suffix(instrument_id, header) == expected


def test_science_array_class_suffix_without_header():
    assert science_array_class_suffix('COS') == '_Spectrum'
    assert science_array_class_suffix('WFPC2') == '_Image'


def _cos_file(tmp_path, obstype):
    """Write a COS file shaped like a NUV flt file: an empty primary HDU, then SCI and DQ
    arrays. Return its path and its open HDUList."""
    primary = fits.PrimaryHDU()
    primary.header['INSTRUME'] = 'COS'
    primary.header['OBSTYPE'] = obstype
    sci = fits.ImageHDU(np.zeros((8, 8), dtype='float32'), name='SCI')
    dq = fits.ImageHDU(np.zeros((8, 8), dtype='int16'), name='DQ')
    path = tmp_path / f'lbg501e8q_flt_{obstype.lower()}.fits'
    fits.HDUList([primary, sci, dq]).writeto(path)
    return (str(path), fits.open(path))


@pytest.mark.parametrize('obstype,expected', [
    ('IMAGING',       'Array_2D_Image'),
    ('SPECTROSCOPIC', 'Array_2D_Spectrum'),
])
def test_fill_hdu_dictionary_cos_arrays(tmp_path, obstype, expected):
    # The arrays of a COS NUV image, e.g., a target acquisition, are images; otherwise
    # they are spectra. The class comes from the primary header, not the extension's.
    (path, hdulist) = _cos_file(tmp_path, obstype)
    with hdulist:
        classes = [fill_hdu_dictionary(hdu, k, 'COS', path,
                                       primary_header=hdulist[0].header)
                   ['data']['data_class'] for (k, hdu) in enumerate(hdulist)]
    assert classes == ['', expected, expected]


def test_fill_hdu_dictionary_reads_obstype_from_primary_hdu(tmp_path):
    # Without a primary header, a data array in the primary HDU is classified by its own
    # header.
    path = tmp_path / 'o4xx01aaq_raw.fits'
    primary = fits.PrimaryHDU(np.zeros((8, 8), dtype='float32'))
    primary.header['OBSTYPE'] = 'IMAGING'
    fits.HDUList([primary]).writeto(path)
    with fits.open(path) as hdulist:
        hdu_dict = fill_hdu_dictionary(hdulist[0], 0, 'STIS', str(path))
    assert hdu_dict['data']['data_class'] == 'Array_2D_Image'


@pytest.mark.parametrize('unit, expected', [('Counts', 'count'), ('K', 'K')])
def test_unit_translator(unit, expected):
    # Units of WFPC/WFPC2 SI_Specific columns and NICMOS epc values
    assert UNIT_TRANSLATOR[unit] == expected
