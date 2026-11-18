##########################################################################################
# tests/test_product_id.py
#
# Tests for product IDs, the last field of a product LID. Files whose extended suffixes
# share one collection (e.g., COS x1dsum and x1dsum1-4) must still get distinct LIDs.
##########################################################################################

import pytest

from hst_pipeline.hst_helper.fs_utils import get_product_id
from hst_pipeline.product_labels.suffix_info import product_id


@pytest.mark.parametrize('ipppssoot,suffix,expected', [
    ('ld8x01010', 'x1dsum', 'ld8x01010'),
    ('ld8x01010', 'x1dsum1', 'ld8x010101'),
    ('ld8x01010', 'x1dsum4', 'ld8x010104'),
    ('lb6a03s1q', 'rawtag_a', 'lb6a03s1q_a'),
    ('lb6a03s1q', 'corrtag_b', 'lb6a03s1q_b'),
    ('n4wl01xdq', 'raw', 'n4wl01xdq'),
    ('u2no0401t', 'c0f', 'u2no0401t'),
])
def test_product_id(ipppssoot, suffix, expected):
    assert product_id(ipppssoot, suffix) == expected


@pytest.mark.parametrize('filename,expected', [
    ('ld8x01010_x1dsum.fits', 'ld8x01010'),
    ('ld8x01010_x1dsum2.fits', 'ld8x010102'),
    ('ld8x01010_x1dsum2.xml', 'ld8x010102'),
    ('ld8x01010_x1dsum2.png', 'ld8x010102'),
    ('lb6a03s1q_flt_a.fits', 'lb6a03s1q_a'),
    ('u2no0401t_d0f.jpg', 'u2no0401t'),
])
def test_get_product_id(filename, expected):
    assert get_product_id(filename) == expected


def test_x1dsum_products_get_distinct_ids():
    ids = {product_id('ld8x01010', s)
           for s in ('x1dsum', 'x1dsum1', 'x1dsum2', 'x1dsum3', 'x1dsum4')}
    assert len(ids) == 5
