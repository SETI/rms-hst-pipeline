##########################################################################################
# tests/test_browse_products.py
#
# Tests for browse products: the JPEG and PNG previews downloaded from MAST, how they are
# filtered, where they are filed, and how data labels refer to them.
##########################################################################################

import os
import xml.etree.ElementTree as ET

import pytest

from hst_pipeline import label_hst_products as label_module
from hst_pipeline import prepare_browse_products as prepare_module
from hst_pipeline.hst_helper import query_utils
from hst_pipeline.label_browse_products import label_browse_collection_directory
from hst_pipeline.product_labels import get_browse_lids
from hst_pipeline.product_labels.suffix_info import (browse_collection_name,
                                                     browse_data_collection_name,
                                                     browse_data_suffix,
                                                     data_to_browse_suffixes,
                                                     is_accepted_browse_suffix)

PDS4_NS = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}


@pytest.mark.parametrize('browse_suffix,instrument_id,expected', [
    ('drz', 'ACS', 'drz'),
    ('x1dsum1', 'COS', 'x1dsum1'),
    ('c0f', 'WFPC2', 'c0m'),
    ('d0f', 'WFPC2', 'd0m'),
    ('c0f', 'WFPC', 'c0f'),
    ('drw', 'WFPC2', 'c0m'),
])
def test_browse_data_suffix(browse_suffix, instrument_id, expected):
    assert browse_data_suffix(browse_suffix, instrument_id) == expected


@pytest.mark.parametrize('data_suffix,instrument_id,expected', [
    ('drz', 'ACS', ['drz']),
    ('c0m', 'WFPC2', ['c0f', 'drw']),
    ('d0m', 'WFPC2', ['d0f']),
    ('c0f', 'FOC', ['c0f']),
])
def test_data_to_browse_suffixes(data_suffix, instrument_id, expected):
    assert data_to_browse_suffixes(data_suffix, instrument_id) == expected


@pytest.mark.parametrize('browse_suffix,instrument_id,expected', [
    ('drz', 'ACS', 'browse_acs_drz'),
    ('x1dsum1', 'COS', 'browse_cos_x1dsum'),
    ('c0f', 'WFPC2', 'browse_wfpc2_c0m'),
    ('drw', 'WFPC2', 'browse_wfpc2_drw'),
])
def test_browse_collection_name(browse_suffix, instrument_id, expected):
    assert browse_collection_name(browse_suffix, instrument_id) == expected


@pytest.mark.parametrize('browse_suffix,instrument_id,expected', [
    ('drz', 'ACS', 'data_acs_drz'),
    ('x1dsum1', 'COS', 'data_cos_x1dsum'),
    ('c0f', 'WFPC2', 'data_wfpc2_c0m'),
    ('drw', 'WFPC2', 'data_wfpc2_c0m'),
])
def test_browse_data_collection_name(browse_suffix, instrument_id, expected):
    assert browse_data_collection_name(browse_suffix, instrument_id) == expected


@pytest.mark.parametrize('filename,expected', [
    ('j9xx01abq_drz.jpg', 'drz'),
    ('ld8x01010_x1dsum1.png', 'x1dsum1'),
    ('j9xx01abq_drz_large.jpg', 'drz_large'),
])
def test_get_suffix_of_preview_uses_everything_after_ipppssoot(filename, expected):
    row = {'productSubGroupDescription': '--', 'productFilename': filename}
    assert query_utils.get_suffix(row) == expected


# Every preview suffix found in a survey of MAST, plus an ACS "_large" preview, which
# depicts no archived FITS product.
@pytest.mark.parametrize('filename,accepted', [
    ('j9xx01abq_raw.jpg', True),
    ('j9xx01abq_flt.jpg', True),
    ('j9xx01abq_flc.jpg', True),
    ('j9xx01abq_drz.jpg', True),
    ('j9xx01abq_drc.jpg', True),
    ('idxx01abq_drz.jpg', True),
    ('n4xx01abq_cal.jpg', True),
    ('n4xx01abq_mos.jpg', True),
    ('w0xx0101t_c0f.jpg', True),
    ('u2xx0101t_c0f.jpg', True),
    ('u2xx0101t_d0f.jpg', True),
    ('u2xx0101t_drw.jpg', True),
    ('x0xx0101t_d0f.jpg', True),
    ('y0xx0101t_c0f.jpg', True),
    ('z0xx0101t_c0f.jpg', True),
    ('o4xx01abq_x1d.png', True),
    ('o4xx01abq_sx1.png', True),
    ('lbxx01abq_x1d.png', True),
    ('lbxx01abq_x1dsum.png', True),
    ('lbxx01abq_x1dsum4.png', True),
    ('j9xx01abq_drz_large.jpg', False),
])
def test_is_accepted_instrument_suffix_for_previews(filename, accepted):
    row = {'obs_id': filename.partition('_')[0],
           'productSubGroupDescription': '--',
           'productFilename': filename}
    assert bool(query_utils.is_accepted_instrument_suffix(row)) is accepted


def test_is_accepted_browse_suffix():
    assert is_accepted_browse_suffix('c0f', 'WFPC2')
    assert is_accepted_browse_suffix('drw', 'WFPC2')
    assert not is_accepted_browse_suffix('drz_large', 'ACS')


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'\xff\xd8\xff\xd9')    # bytes need not form a valid image here


def _basename_dict(fullpath, instrument_id):
    ipppssoot, _, suffix = os.path.basename(fullpath).partition('.')[0].partition('_')
    return {'fullpath': fullpath, 'ipppssoot': ipppssoot, 'suffix': suffix,
            'instrument_id': instrument_id,
            'hst_dictionary': {'hst_proposal_id': 5167}}


def test_get_browse_lids_finds_preview_beside_fits(tmp_path):
    fits = str(tmp_path / 'j9xx01abq_drz.fits')
    _touch(fits)
    assert get_browse_lids(_basename_dict(fits, 'ACS')) == []

    _touch(str(tmp_path / 'j9xx01abq_drz.jpg'))
    _touch(str(tmp_path / 'j9xx01abq_flt.jpg'))     # a different product's preview
    assert get_browse_lids(_basename_dict(fits, 'ACS')) == [
        'urn:nasa:pds:hst_05167:browse_acs_drz:j9xx01abq'
    ]


def test_get_browse_lids_keeps_extended_suffix_in_lid(tmp_path):
    for suffix, prod_id in (('x1dsum', 'ld8x01010'), ('x1dsum3', 'ld8x010103')):
        fits = str(tmp_path / f'ld8x01010_{suffix}.fits')
        _touch(fits)
        _touch(str(tmp_path / f'ld8x01010_{suffix}.png'))
        assert get_browse_lids(_basename_dict(fits, 'COS')) == [
            f'urn:nasa:pds:hst_05167:browse_cos_x1dsum:{prod_id}'
        ]


def test_get_browse_lids_finds_preview_in_sibling_browse_collection(tmp_path):
    # The staging and bundle layout: <program>/<collection>/visit_<ss>/<file>.
    fits = str(tmp_path / 'hst_05167' / 'data_wfpc2_c0m' / 'visit_04' / 'u2xx0401t_c0m.fits')
    _touch(fits)
    assert get_browse_lids(_basename_dict(fits, 'WFPC2')) == []

    for collection, name in (('browse_wfpc2_c0m', 'u2xx0401t_c0f.jpg'),
                             ('browse_wfpc2_drw', 'u2xx0401t_drw.jpg'),
                             ('browse_wfpc2_c0m', 'u2xx0402t_c0f.jpg')):  # other obs
        _touch(str(tmp_path / 'hst_05167' / collection / 'visit_04' / name))
    _touch(str(tmp_path / 'hst_05167' / 'browse_wfpc2_c0m' / 'visit_05' /
               'u2xx0401t_c0f.jpg'))                                        # other visit
    assert get_browse_lids(_basename_dict(fits, 'WFPC2')) == [
        'urn:nasa:pds:hst_05167:browse_wfpc2_c0m:u2xx0401t',
        'urn:nasa:pds:hst_05167:browse_wfpc2_drw:u2xx0401t',
    ]


def test_get_browse_lids_maps_waivered_and_drizzled_wfpc2_previews(tmp_path):
    fits = str(tmp_path / 'u2xx0101t_c0m.fits')
    _touch(fits)
    _touch(str(tmp_path / 'u2xx0101t_c0f.jpg'))
    _touch(str(tmp_path / 'u2xx0101t_drw.jpg'))
    assert get_browse_lids(_basename_dict(fits, 'WFPC2')) == [
        'urn:nasa:pds:hst_05167:browse_wfpc2_c0m:u2xx0101t',
        'urn:nasa:pds:hst_05167:browse_wfpc2_drw:u2xx0101t',
    ]


def test_prepare_browse_products_files_previews_under_mast_names(tmp_path, monkeypatch):
    def fake_program_dir_path(proposal_id, visit=None, root_dir='staging'):
        path = str(tmp_path / f'hst_{int(proposal_id):05d}')
        return path + (f'/visit_{visit}' if visit else '')

    monkeypatch.setattr(prepare_module, 'get_program_dir_path', fake_program_dir_path)
    monkeypatch.setattr(label_module, 'get_program_dir_path', fake_program_dir_path)

    download_dir = tmp_path / 'hst_05167' / 'visit_01' / 'mastDownload' / 'HST'
    for name in ('j9xx01abq/j9xx01abq_drz.jpg',
                 'u2xx0101t/u2xx0101t_c0f.jpg',
                 'u2xx0101t/u2xx0101t_drw.jpg',
                 'j9xx01abq/j9xx01abq_drz_large.jpg',
                 'ld8x01010/ld8x01010_x1dsum1.png'):
        _touch(str(download_dir / name))

    # File the previews, then label them, as the pipeline does.
    prepare_module.prepare_browse_products(5167, '01')
    label_module.label_hst_visit(5167, '01')

    staging = tmp_path / 'hst_05167'
    assert sorted(os.listdir(staging / 'browse_acs_drz' / 'visit_01')) == [
        'j9xx01abq_drz.jpg', 'j9xx01abq_drz.xml'
    ]
    assert sorted(os.listdir(staging / 'browse_wfpc2_c0m' / 'visit_01')) == [
        'u2xx0101t_c0f.jpg', 'u2xx0101t_c0f.xml'
    ]
    assert sorted(os.listdir(staging / 'browse_wfpc2_drw' / 'visit_01')) == [
        'u2xx0101t_drw.jpg', 'u2xx0101t_drw.xml'
    ]
    assert sorted(os.listdir(staging / 'browse_cos_x1dsum' / 'visit_01')) == [
        'ld8x01010_x1dsum1.png', 'ld8x01010_x1dsum1.xml'
    ]

    # No collection name carries the old "mast" prefix.
    assert not any('mast' in name for name in os.listdir(staging))

    # The "_large" preview depicts no archived FITS product, so it is not filed; it is the
    # only file left in the inbox.
    filed = [name for (dirpath, _, names) in os.walk(staging) for name in names
             if not dirpath.startswith(str(staging / 'visit_01'))]
    assert 'j9xx01abq_drz_large.jpg' not in filed
    inbox_files = [name for (_, _, names) in os.walk(staging / 'visit_01')
                   for name in names]
    assert inbox_files == ['j9xx01abq_drz_large.jpg']

    # Both WFPC2 browse labels point back at the data product with the unwaivered suffix.
    for collection, preview in (('browse_wfpc2_c0m', 'u2xx0101t_c0f.jpg'),
                                ('browse_wfpc2_drw', 'u2xx0101t_drw.jpg')):
        label = preview.rpartition('.')[0] + '.xml'
        root = ET.parse(staging / collection / 'visit_01' / label)
        file_names = [e.text
                      for e in root.iterfind('.//pds:File/pds:file_name', PDS4_NS)]
        assert file_names == [preview]
        lids = [e.text for e in root.iterfind('.//pds:logical_identifier', PDS4_NS)]
        assert lids == [f'urn:nasa:pds:hst_05167:{collection}:u2xx0101t']
        refs = [e.text for e in root.iterfind('.//pds:lidvid_reference', PDS4_NS)]
        assert refs == ['urn:nasa:pds:hst_05167:data_wfpc2_c0m:u2xx0101t::1.0']


def test_browse_labels_are_named_after_each_image(tmp_path):
    collection_dir = tmp_path / 'browse_cos_x1dsum'
    visit_dir = collection_dir / 'visit_01'
    for name in ('ld8x01010_x1dsum.png', 'ld8x01010_x1dsum1.png'):
        _touch(str(visit_dir / name))

    assert label_browse_collection_directory(str(collection_dir), 5167) == 2
    assert sorted(os.listdir(visit_dir)) == [
        'ld8x01010_x1dsum.png', 'ld8x01010_x1dsum.xml',
        'ld8x01010_x1dsum1.png', 'ld8x01010_x1dsum1.xml',
    ]
    # The previews share a collection but get distinct LIDs, each linked to the matching
    # x1dsum data product.
    for stem, prod_id in (('ld8x01010_x1dsum', 'ld8x01010'),
                          ('ld8x01010_x1dsum1', 'ld8x010101')):
        root = ET.parse(visit_dir / f'{stem}.xml').getroot()
        file_names = [e.text
                      for e in root.iterfind('.//pds:File/pds:file_name', PDS4_NS)]
        assert file_names == [f'{stem}.png']
        lids = [e.text for e in root.iterfind('.//pds:logical_identifier', PDS4_NS)]
        assert lids == [f'urn:nasa:pds:hst_05167:browse_cos_x1dsum:{prod_id}']
        refs = [e.text for e in root.iterfind('.//pds:lidvid_reference', PDS4_NS)]
        assert refs == [f'urn:nasa:pds:hst_05167:data_cos_x1dsum:{prod_id}::1.0']
