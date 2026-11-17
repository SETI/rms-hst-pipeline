##########################################################################################
# tests/test_target_support.py
#
# Tests for the adapter that maps rms-hst-targets results onto the pipeline's target
# identification tuples.
##########################################################################################

import os

import pytest

from hst_pipeline import target_support
from hst_pipeline.target_support import hst_target_identifications

SPT_PATH = os.path.join(os.path.dirname(__file__), 'golden_copies', 'hst_07885',
                        'visit_01', 'mastDownload', 'HST', 'n4wl01xdq',
                        'n4wl01xdq_spt.fits')


def fake_targets(monkeypatch, dicts):
    """Make the adapter see `dicts` as the identified bodies.

    _complete_target is replaced by the identity function, so each dictionary must
    already carry the completed fields the adapter reads.
    """
    monkeypatch.setattr(target_support, 'identify_target_dicts',
                        lambda headers, logger=None: list(dicts))
    monkeypatch.setattr(target_support, '_complete_target', lambda target: target)


def completed(**overrides):
    """A completed target dictionary, with any field overridden."""
    target = {'title':       'Uranus',
              'alt_titles':  ['NAIF ID 799'],
              'type_name':   'Planet',
              'description': [],
              'lid':         'urn:nasa:pds:context:target:planet.uranus'}
    target.update(overrides)
    return target


def test_maps_fields_onto_the_tuple(monkeypatch):
    fake_targets(monkeypatch, [completed()])
    assert hst_target_identifications({}, 'x_spt.fits') == [
        ('Uranus', ['NAIF ID 799'], 'Planet', [], 'urn:nasa:pds:context:target:planet.uranus')
    ]


def test_returns_one_tuple_per_body(monkeypatch):
    fake_targets(monkeypatch, [completed(),
                               completed(title='Ariel',
                                         lid='urn:nasa:pds:context:target:satellite.ariel',
                                         type_name='Satellite',
                                         alt_titles=[])])
    records = hst_target_identifications({}, 'x_spt.fits')
    assert [r[0] for r in records] == ['Uranus', 'Ariel']
    assert [r[2] for r in records] == ['Planet', 'Satellite']


def test_description_is_passed_through(monkeypatch):
    fake_targets(monkeypatch, [completed(description=['line one', 'line two'])])
    assert hst_target_identifications({}, 'x_spt.fits')[0][3] == ['line one', 'line two']


def test_lists_are_copied_not_aliased(monkeypatch):
    """The caller must not be able to mutate the dictionaries rms-hst-targets returned."""
    target = completed()
    fake_targets(monkeypatch, [target])
    name, alts, _, description, _ = hst_target_identifications({}, 'x_spt.fits')[0]
    alts.append('mutated')
    description.append('mutated')
    assert target['alt_titles'] == ['NAIF ID 799']
    assert target['description'] == []


def test_no_targets_gives_empty_list(monkeypatch):
    fake_targets(monkeypatch, [])
    assert hst_target_identifications({}, 'x_spt.fits') == []


def test_identification_failure_propagates_as_value_error(monkeypatch):
    """The call site in product_labels catches ValueError, so failures must stay one."""
    from targets import NotPlanetaryError, TargetIdentificationFailure

    assert issubclass(TargetIdentificationFailure, ValueError)
    assert issubclass(NotPlanetaryError, ValueError)

    def boom(headers, logger=None):
        raise TargetIdentificationFailure('no target')

    monkeypatch.setattr(target_support, 'identify_target_dicts', boom)
    with pytest.raises(ValueError):
        hst_target_identifications({}, 'x_spt.fits')


@pytest.mark.targets_data
def test_real_spt_header_identifies_uranus():
    """End-to-end against a real SPT header.

    This is the canary on rms-hst-targets' private _complete_target(): if its field names
    change, this fails while the mocked tests above keep passing. It needs that package's
    committed caches, so it is deselected wherever they are unavailable.
    """
    from astropy.io import fits

    with fits.open(SPT_PATH) as hdulist:
        header = hdulist[0].header

    records = hst_target_identifications(header, SPT_PATH)
    assert len(records) == 1
    name, alts, body_type, description, lid = records[0]
    assert name == 'Uranus'
    assert body_type == 'Planet'
    assert lid == 'urn:nasa:pds:context:target:planet.uranus'
    assert 'NAIF ID 799' in alts

##########################################################################################
