##########################################################################################
# tests/test_hst_dictionary_support.py
#
# Tests for fill_hst_dictionary, using small synthetic FITS headers, and for the plate
# scale nilReason.
##########################################################################################

from astropy.io import fits
import pytest

from hst_pipeline.product_labels.hst_dictionary_support import fill_hst_dictionary
from hst_pipeline.product_labels.suffix_info import plate_scale_nil_reason


def _hdulist(**keywords):
    """Return an HDUList whose primary header holds the given keywords."""
    hdu = fits.PrimaryHDU()
    for (key, value) in keywords.items():
        hdu.header[key] = value
    return fits.HDUList([hdu])


@pytest.mark.parametrize('ref_keywords,spt_keywords,expected', [
    # Science file: OPT_ELEM wins
    ({'OPT_ELEM': 'G130M', 'DETECTOR': 'FUV'}, {'SPEC_1': 'G140L'}, 'G130M'),
    # NUV imaging: the mirror is the optical element
    ({'OPT_ELEM': 'MIRRORA', 'DETECTOR': 'NUV'}, {'SPEC_1': 'MIRRORA'}, 'MIRRORA'),
    # Ancillary file without OPT_ELEM (jit, trl, asn...): use the spt file's SPEC_1
    ({}, {'SPEC_1': 'G140L'}, 'G140L'),
    # Neither available
    ({}, {}, 'UNK'),
])
def test_cos_filter_name(ref_keywords, spt_keywords, expected):
    ref = _hdulist(INSTRUME='COS', **ref_keywords)
    spt = _hdulist(INSTRUME='COS', **spt_keywords)
    hst_dictionary = fill_hst_dictionary(ref, spt, '/tmp/ld8x01doq_x1d.fits')
    assert hst_dictionary['filter_name'] == expected


@pytest.mark.parametrize('instrument_id,expected', [
    ('COS', 'inapplicable'),
    ('FGS', 'inapplicable'),
    ('FOS', 'inapplicable'),
    ('GHRS', 'inapplicable'),
    ('HSP', 'inapplicable'),
    ('ACS', 'missing'),
    ('STIS', 'missing'),
    ('WFPC2', 'missing'),
])
def test_plate_scale_nil_reason(instrument_id, expected):
    assert plate_scale_nil_reason(instrument_id) == expected
