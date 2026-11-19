##########################################################################################
# tests/test_wavelength_ranges.py
#
# Tests for wavelength_ranges, the Science_Facets wavelength ranges of a product.
##########################################################################################

import pytest

from hst_pipeline.product_labels.wavelength_ranges import wavelength_ranges


@pytest.mark.parametrize('detector_ids,filter_name,expected', [
    # ACS/SBC is a far-UV detector, so every filter and prism is UV, including the
    # prisms, whose names hold no wavelength
    (['SBC'], 'PR110L', ['Ultraviolet']),
    (['SBC'], 'PR130L', ['Ultraviolet']),
    (['SBC'], 'F115LP', ['Ultraviolet']),
    (['SBC'], 'F122M', ['Ultraviolet']),
    # Other ACS detectors use the filter
    (['HRC'], 'F330W', ['Ultraviolet']),
])
def test_acs_wavelength_ranges(detector_ids, filter_name, expected):
    assert wavelength_ranges('ACS', detector_ids, filter_name) == expected


def test_unknown_prism_without_detector_rule_raises():
    # Without a detector rule, a filter name with no wavelength cannot be resolved; the
    # labeler reports this as "Undetermined wavelength range".
    with pytest.raises((ValueError, KeyError)):
        wavelength_ranges('ACS', ['HRC'], 'PR130L')
