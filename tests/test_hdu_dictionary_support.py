##########################################################################################
# tests/test_hdu_dictionary_support.py
#
# Tests for the data classes that hdu_dictionary_support assigns to FITS data objects.
##########################################################################################

import numpy as np
import pdslogger
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


def _epc_table(path, unit):
    """Write a NICMOS-epc-like file whose one table has a VALUE column in the given unit;
    return the logger's warning count and the column's unit in the label."""
    table = fits.BinTableHDU.from_columns([
        fits.Column(name='TIME', format='D', unit='MJD', array=[54309.18]),
        fits.Column(name='VALUE', format='D', unit=unit, array=[1207.6]),
    ], name='MNCORSPD')
    path.parent.mkdir()
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    logger = pdslogger.PdsLogger(f'test.epc-units-{path.parent.name}-{path.stem}')
    logger.open('test')
    with fits.open(path) as hdulist:
        hdu_dict = fill_hdu_dictionary(hdulist[1], 1, 'NICMOS', str(path), logger)
    return (logger.close()[2], hdu_dict['data']['units'][1])


@pytest.mark.parametrize('unit, expected', [('rps', 'Hz'), ('A', 'A')])
def test_epc_units(tmp_path, unit, expected):
    # n9z702d2j_epc.fits (11118): cooling-system rotation speeds and a current
    assert _epc_table(tmp_path / unit / 'n9z702d2j_epc.fits', unit) == (0, expected)


def test_ampere_is_recognized_only_in_epc_files(tmp_path):
    # Elsewhere, "A" may mean Angstroms, so it still warns
    assert _epc_table(tmp_path / 'spt' / 'n9z702d2q_spt.fits', 'A') == (1, 'A')
