##########################################################################################
# tests/test_associations.py
#
# Tests for reading ASN tables and for the prior suffixes that link each product to the
# files it was made from.
##########################################################################################

from astropy.io import fits

from hst_pipeline.product_labels import read_associations
from hst_pipeline.product_labels import suffix_info


def _asn(rows):
    names = [name for (name, _) in rows]
    types = [memtype for (_, memtype) in rows]
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='MEMNAME', format='14A', array=names),
        fits.Column(name='MEMTYPE', format='14A', array=types),
        fits.Column(name='MEMPRSNT', format='L', array=[True] * len(rows)),
    ])
    return fits.HDUList([fits.PrimaryHDU(), hdu])


def test_dither_product_gets_every_exposure():
    # j9mm03010_asn.fits: a dithered CR-SPLIT association
    hdulist = _asn([('J9MM03NWQ', 'EXP-CR1'), ('J9MM03NXQ', 'EXP-CR1'),
                    ('J9MM03O2Q', 'EXP-CR2'), ('J9MM03O3Q', 'EXP-CR2'),
                    ('J9MM03010', 'PROD-DTH'), ('J9MM03011', 'PROD-CR1'),
                    ('J9MM03012', 'PROD-CR2')])
    assert read_associations(hdulist, 'j9mm03010_asn.fits') == {
        'j9mm03010': [('j9mm03nwq', 'EXP'), ('j9mm03nxq', 'EXP'),
                      ('j9mm03o2q', 'EXP'), ('j9mm03o3q', 'EXP')],
        'j9mm03011': [('j9mm03nwq', 'EXP'), ('j9mm03nxq', 'EXP')],
        'j9mm03012': [('j9mm03o2q', 'EXP'), ('j9mm03o3q', 'EXP')],
    }


def test_dither_product_with_its_own_exposures():
    hdulist = _asn([('IB1A01A1Q', 'EXP-DTH'), ('IB1A01A2Q', 'EXP-DTH'),
                    ('IB1A01010', 'PROD-DTH')])
    assert read_associations(hdulist, 'ib1a01010_asn.fits') == {
        'ib1a01010': [('ib1a01a1q', 'EXP-DTH'), ('ib1a01a2q', 'EXP-DTH')],
    }


def test_prior_suffixes_cannot_be_altered():
    # The labeler once intersected this set in place, emptying it for every later
    # IPPPSSOOT of the visit, so a later drz lost its references to its flt files.
    prior = suffix_info.get_prior_suffixes('drz', 'ACS')
    assert prior == {'flt'}
    prior &= set()
    assert suffix_info.get_prior_suffixes('drz', 'ACS') == {'flt'}

