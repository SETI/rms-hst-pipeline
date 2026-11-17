##########################################################################################
# tests/test_browse_previews.py
#
# Tests for picmaker-generated browse products: the versions configs, the tiers they
# produce, file naming, and the order of files in browse labels.
##########################################################################################

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hst_pipeline.generate_browse_previews import (expected_browse_jpg_names,
                                                   generate_browse_previews)
from hst_pipeline.label_browse_products import (_browse_sort_key,
                                                label_browse_collection_directory)
from hst_pipeline.product_labels.suffix_info import (BROWSE_SUFFIX_INFO,
                                                     PICMAKER_BROWSE_SUFFIXES,
                                                     PICMAKER_BROWSE_TIERS)

CONFIG_DIR = (Path(__file__).resolve().parent.parent
              / 'src' / 'hst_pipeline' / 'picmaker_configs')
CONFIG_FILES = sorted(CONFIG_DIR.glob('*_previews.txt'))
NICMOS_RAW = (Path(__file__).resolve().parent / 'golden_copies' / 'hst_07885'
              / 'visit_01' / 'mastDownload' / 'HST' / 'n4wl01xdq'
              / 'n4wl01xdq_raw.fits')
PDS4_NS = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}


def test_config_files_found():
    assert len(CONFIG_FILES) == 6


@pytest.mark.parametrize('config', CONFIG_FILES, ids=lambda p: p.name)
def test_config_has_thumb_and_unsuffixed_full_size_only(config):
    lines = [line.split() for line in config.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    thumb, full = lines
    assert thumb[thumb.index('--suffix') + 1] == '_thumb'
    assert '--suffix' not in full


@pytest.mark.parametrize('config', CONFIG_FILES, ids=lambda p: p.name)
def test_config_has_no_tint(config):
    assert '--tint' not in config.read_text()


def test_tiers():
    assert PICMAKER_BROWSE_TIERS == {'thumb': '_thumb', 'full': ''}


def test_expected_browse_jpg_names():
    assert expected_browse_jpg_names('/a/b/n4wl01xdq_raw.fits') == [
        'n4wl01xdq_thumb.jpg', 'n4wl01xdq.jpg'
    ]


def test_browse_suffix_info_lists_generated_tiers():
    for instrument_id, fits_suffixes in PICMAKER_BROWSE_SUFFIXES.items():
        for fits_suffix in fits_suffixes:
            generated = [info.suffix
                         for info in BROWSE_SUFFIX_INFO[instrument_id, fits_suffix]
                         if info.is_locally_generated]
            assert generated == ['thumb', 'full']


def test_browse_sort_key_puts_thumb_before_full_size():
    names = ['n4wl01xdq.jpg', 'n4wl01xdq_thumb.jpg']
    assert sorted(names, key=_browse_sort_key) == ['n4wl01xdq_thumb.jpg',
                                                   'n4wl01xdq.jpg']


def test_generate_browse_previews_writes_untinted_thumb_and_full_size(tmp_path):
    out_dir = tmp_path / 'browse_generated_nicmos_raw' / 'visit_01'
    assert generate_browse_previews(NICMOS_RAW, out_dir, 'NICMOS', 'raw')
    assert sorted(os.listdir(out_dir)) == ['n4wl01xdq.jpg', 'n4wl01xdq_thumb.jpg']

    for name in os.listdir(out_dir):
        with Image.open(out_dir / name) as image:
            rgb = np.asarray(image.convert('RGB')).astype(int)
        # Untinted means every pixel is gray: all three channels equal.
        assert np.array_equal(rgb[..., 0], rgb[..., 1])
        assert np.array_equal(rgb[..., 1], rgb[..., 2])

    with Image.open(out_dir / 'n4wl01xdq_thumb.jpg') as thumb:
        assert max(thumb.size) == 100


def test_browse_label_lists_thumb_then_full_size(tmp_path):
    collection_dir = tmp_path / 'browse_generated_nicmos_raw'
    visit_dir = collection_dir / 'visit_01'
    visit_dir.mkdir(parents=True)
    for name in ('n4wl01xdq.jpg', 'n4wl01xdq_thumb.jpg'):
        Image.new('L', (4, 4)).save(visit_dir / name)

    assert label_browse_collection_directory(str(collection_dir), 7885) == 1

    root = ET.parse(visit_dir / 'n4wl01xdq.xml').getroot()
    file_names = [e.text for e in root.iterfind('.//pds:File/pds:file_name', PDS4_NS)]
    assert file_names == ['n4wl01xdq_thumb.jpg', 'n4wl01xdq.jpg']
