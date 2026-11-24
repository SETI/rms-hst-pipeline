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
    target = {'name': '16 Psyche', 'type': 'Asteroid',
              'alternate_designations': ['(16) Psyche', '16 Psyche', 'NAIF ID 2000016'],
              'formatted_name': '16_psyche', 'formatted_type': 'asteroid',
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
        # The name and type that the product labels use, and the target label's version
        'urn:nasa:pds:context:target:asteroid.16_psyche::1.0':
            (['16 Psyche', '(16) Psyche', 'NAIF ID 2000016'], ['Asteroid']),
    }


def test_target_template_version_and_names():
    # The JSON gives the version, name, and type that TARGET_LABEL.lblx writes.
    with open(os.path.join(TEMPLATE_DIR, 'TARGET_LABEL.lblx')) as f:
        text = f.read()
    assert f'<version_id>{ctl.TARGET_LABEL_VID}</version_id>' in text
    assert "<name>$target['name']$</name>" in text
    assert "<type>$target['type']$</type>" in text


def test_create_target_label_once_per_lid(monkeypatch):
    # A target identified twice, with different alternate designations, gets one label.
    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'<a href="planet.uranus_1.3.xml">planet.uranus_1.3.xml</a>'

    monkeypatch.setattr(ctl.urllib.request, 'urlopen', lambda url: _Response())
    labels = []
    monkeypatch.setattr(ctl, 'create_collection_label',
                        lambda p_id, col, data, filename, *args: labels.append(filename))
    added = []
    monkeypatch.setattr(ctl, 'create_tmp_context_json',
                        lambda p_id, data, targs: added.extend(targs))

    prefix = 'urn:nasa:pds:context:target:'
    system = {'name': 'Uranus System', 'type': 'Planetary System',
              'alternate_designations': [], 'lid': prefix + 'planetary_system.uranus_system'}
    system2 = {**system, 'alternate_designations': ['Uranian System']}
    uranus = {'name': 'Uranus', 'type': 'Planet', 'alternate_designations': [],
              'lid': prefix + 'planet.uranus'}
    data_dict = {'target_identifications': [system, uranus, system2],
                 'label_date': '2026-11-24'}
    ctl.create_target_label(12245, data_dict, None)

    assert labels == ['planetary_system.uranus_system_1.0.lblx']
    assert added == [system]


@pytest.mark.parametrize('template', ['PRODUCT_LABEL.lblx',
                                      'PRODUCT_COLLECTION_LABEL.lblx',
                                      'BUNDLE_LABEL.lblx'])
def test_templates_use_the_investigation_name_and_type(template):
    # If a template's wording changes, the JSON must change with it.
    with open(os.path.join(TEMPLATE_DIR, template)) as f:
        text = f.read()
    assert f'<name>{ctl.INVESTIGATION_NAME_FMT.format("$prop_id$")}</name>' in text
    assert f'<type>{ctl.INVESTIGATION_TYPE}</type>' in text


def test_product_template_uses_the_host_and_instrument_types():
    with open(os.path.join(TEMPLATE_DIR, 'PRODUCT_LABEL.lblx')) as f:
        text = f.read()
    assert f'<name>{ctl.HOST_NAME}</name>' in text
    assert f'<type>{ctl.HOST_TYPE}</type>' in text
    assert f'<type>{ctl.INSTRUMENT_TYPE}</type>' in text
