##########################################################################################
# tests/test_associations.py
#
# Tests for reading ASN tables, for the prior suffixes that link each product to the
# files it was made from, for the combined products that give an association's
# metadata, and for the chips of its ACS/WFC images, which come from its exposures.
##########################################################################################

from astropy.io import fits

from hst_pipeline.product_labels import association_observation_id
from hst_pipeline.product_labels import fill_hst_dictionary
from hst_pipeline.product_labels import association_product_dicts
from hst_pipeline.product_labels import read_associations
from hst_pipeline.product_labels import use_source_detector_ids
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


def _ipppssoot_dict(suffixes, associates):
    ipppssoot_dict = {suffix: {'suffix': suffix} for suffix in suffixes}
    ipppssoot_dict['all_suffixes'] = set(suffixes)
    ipppssoot_dict['associates'] = associates
    return ipppssoot_dict


def test_association_products():
    # j9mm03051: its drz, not its exposures' raw files, gives its metadata
    exposures = [('j9mm03obq', 'EXP-CRJ'), ('j9mm03ocq', 'EXP-CRJ')]
    ipppssoot_dict = _ipppssoot_dict(['asn', 'drz', 'drc', 'jif', 'jit', 'spt', 'trl'],
                                     exposures)
    assert [d['suffix'] for d in association_product_dicts(ipppssoot_dict)] == ['drz']

    # A STIS association has its own raw file
    ipppssoot_dict = _ipppssoot_dict(['asn', 'raw', 'crj', 'sx2', 'spt', 'trl'],
                                     exposures)
    assert [d['suffix'] for d in association_product_dicts(ipppssoot_dict)] == ['raw']


def test_no_association_products():
    # j9mm03011, a CR-SPLIT product with no file of its own beyond ancillary ones
    exposures = [('j9mm03nwq', 'EXP'), ('j9mm03nxq', 'EXP')]
    assert association_product_dicts(_ipppssoot_dict(['jif', 'jit', 'spt', 'trl'],
                                                     exposures)) == []
    # An exposure, which has no associates
    assert association_product_dicts(_ipppssoot_dict(['raw', 'flt', 'spt'], [])) == []


def test_association_observation_id():
    exposures = [('lbg501eaq', 'EXP-FP'), ('lbg501ecq', 'EXP-FP')]
    # COS x1dsum: ROOTNAME names an exposure, ASN_ID the association
    assert association_observation_id('lbg501eaq', 'LBG501010', exposures) == 'lbg501010'
    # ACS drz: ROOTNAME names the product
    assert association_observation_id('j9mm03051', '', exposures) == 'j9mm03051'
    assert association_observation_id('j9mm03010', 'J9MM03010', exposures) == 'j9mm03010'


def _wfc_file(path, ccdchips, aperture='WFC2-2K'):
    primary = fits.PrimaryHDU()
    for (key, value) in [('INSTRUME', 'ACS'), ('DETECTOR', 'WFC'), ('APERTURE', aperture),
                         ('PROPOSID', 10782)]:
        primary.header[key] = value
    hdus = [primary]
    for (k, ccdchip) in enumerate(ccdchips):
        hdu = fits.ImageHDU(name='SCI', ver=k+1)
        hdu.header['CCDCHIP'] = ccdchip
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path)
    return str(path)


def test_use_source_detector_ids(tmp_path):
    # j9mm03051: its drz has CCDCHIP = -999, which would mean both chips; its exposures
    # are on WFC2 only
    drz = _wfc_file(tmp_path / 'j9mm03051_drz.fits', [-999])
    raws = [_wfc_file(tmp_path / f'j9mm03{x}_raw.fits', [2]) for x in ('obq', 'ocq')]
    with fits.open(drz) as spt:
        hst_dictionary = fill_hst_dictionary(spt, spt, drz)
        assert hst_dictionary['detector_ids'] == ['WFC1', 'WFC2']
        assert hst_dictionary['detector_id_vs_extver'] == {1: ''}

        use_source_detector_ids(hst_dictionary, raws, spt)
        assert hst_dictionary['detector_ids'] == ['WFC2']
        assert hst_dictionary['detector_id_vs_extver'] == {}

        # A full-frame exposure covers both chips
        full = _wfc_file(tmp_path / 'j9mm03oeq_raw.fits', [2, 1], aperture='WFC')
        hst_dictionary = fill_hst_dictionary(spt, spt, drz)
        use_source_detector_ids(hst_dictionary, [full], spt)
        assert hst_dictionary['detector_ids'] == ['WFC1', 'WFC2']
