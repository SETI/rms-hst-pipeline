##########################################################################################
# tests/test_create_target_label.py
#
# Tests for the temporary context-products JSON passed to the validator: every context
# product must carry the name and type that the labels use for it.
##########################################################################################

import json
import os

import pytest

import hst_pipeline
from hst_pipeline import create_target_label as ctl
from hst_pipeline.hst_helper import fs_utils

TEMPLATE_DIR = os.path.join(os.path.dirname(hst_pipeline.__file__), 'templates')


def _read_json(tmp_path, proposal_id):
    path = tmp_path / f'hst_{proposal_id:05}-deliverable' / \
           f'tmp-context-products-{proposal_id:05}.json'
    return {entry['lidvid']: (entry['name'], entry['type'])
            for entry in json.loads(path.read_text())['Product_Context']}


def test_context_json_names_and_types(tmp_path, monkeypatch):
    monkeypatch.setitem(fs_utils.__dict__, 'HST_DIR', {'bundles': str(tmp_path)})
    target = {'formatted_name': '16_psyche', 'formatted_type': 'asteroid',
              'lid': 'urn:nasa:pds:context:target:asteroid.16_psyche'}
    ctl.create_tmp_context_json(5167, {'inst_id_li': ['GHRS', 'WFPC2']}, [target])

    assert _read_json(tmp_path, 5167) == {
        'urn:nasa:pds:context:investigation:individual.hst_05167::1.0':
            (['HST observing program 5167'], ['Individual Investigation']),
        'urn:nasa:pds:context:instrument_host:spacecraft.hst::1.0':
            (['Hubble Space Telescope'], ['Host']),
        'urn:nasa:pds:context:instrument:hst.ghrs::1.0':
            (['HST Goddard High Resolution Spectrograph'], ['Instrument']),
        'urn:nasa:pds:context:instrument:hst.wfpc2::1.0':
            (['Wide Field Planetary Camera 2 for HST'], ['Instrument']),
        'urn:nasa:pds:context:target:asteroid.16_psyche::1.1':
            (['16_psyche'], ['asteroid']),
    }


@pytest.mark.parametrize('template', ['PRODUCT_LABEL.xml',
                                      'PRODUCT_COLLECTION_LABEL.xml',
                                      'BUNDLE_LABEL.xml'])
def test_templates_use_the_investigation_name_and_type(template):
    # If a template's wording changes, the JSON must change with it.
    with open(os.path.join(TEMPLATE_DIR, template)) as f:
        text = f.read()
    assert f'<name>{ctl.INVESTIGATION_NAME_FMT.format("$prop_id$")}</name>' in text
    assert f'<type>{ctl.INVESTIGATION_TYPE}</type>' in text


def test_product_template_uses_the_host_and_instrument_types():
    with open(os.path.join(TEMPLATE_DIR, 'PRODUCT_LABEL.xml')) as f:
        text = f.read()
    assert f'<name>{ctl.HOST_NAME}</name>' in text
    assert f'<type>{ctl.HOST_TYPE}</type>' in text
    assert f'<type>{ctl.INSTRUMENT_TYPE}</type>' in text
