import pytest
from HST.target_identifications import lids, roman, standard_bodies, minor_planets
from HST.target_identifications.comets import comet_identifications
import HST.target_identifications as ti

# lids.py
def test_lids_clean_basic():
    assert lids.clean('Earth') == 'urn:nasa:pds:context:target:earth'
    assert lids.clean('C/1995 O1 (Hale-Bopp)') == 'urn:nasa:pds:context:target:c1995_o1_hale-bopp'  # No underscore after c
    assert lids.clean('Éarth') == 'urn:nasa:pds:context:target:earth'
    assert lids.clean('target:Pluto') == 'urn:nasa:pds:context:target:pluto'

# roman.py
def test_roman_int_to_roman_and_back():
    for i, r in [(1, 'I'), (4, 'IV'), (9, 'IX'), (58, 'LVIII'), (1994, 'MCMXCIV')]:
        assert roman.int_to_roman(i) == r
        assert roman.roman_to_int(r) == i

# standard_bodies.py
def test_standard_body_identifications_basic():
    # Known planet
    result = standard_bodies.standard_body_identifications('Earth')
    assert any('Earth' in tid[0] for tid in result)
    # Known satellite
    result = standard_bodies.standard_body_identifications('Moon')
    assert any('Moon' in tid[0] for tid in result)
    # Include parent
    result = standard_bodies.standard_body_identifications('Moon', include=['parent'])
    assert any('Earth' in tid[0] for tid in result)
    # Include satellites
    result = standard_bodies.standard_body_identifications('Jupiter', include=['satellites'])
    assert any('Io' in tid[0] for tid in result)
    # Error on unknown
    with pytest.raises(KeyError):
        standard_bodies.standard_body_identifications('NotAPlanet')

# minor_planets.py
def test_minor_planet_identifications_basic(monkeypatch):
    # Patch get_mpc_info to return a tuple as expected
    class FakeInfo:
        def __init__(self):
            self.name = 'Chiron'
            self.alt_designations = ['2060']
            self.body_type = 'Centaur'
            self.description = ['A centaur.']
            self.lid = 'chiron'
        def target_identifications(self):
            return [(self.name, self.alt_designations, self.body_type, self.description, self.lid)]
    monkeypatch.setattr(minor_planets, 'get_mpc_info', lambda key: (['Chiron'], 0, 0, 0, 0))
    result = minor_planets.minor_planet_identifications('2060')
    assert any('Chiron' in tid[0] for tid in result)
    # Error on unknown
    monkeypatch.setattr(minor_planets, 'get_mpc_info', lambda key: None)
    with pytest.raises(KeyError):
        minor_planets.minor_planet_identifications('NotAMinorPlanet')

# comets/__init__.py
def test_comet_identifications_basic(monkeypatch):
    # Patch identify_comet to return a fake CometInfo
    class FakeComet:
        mp_number = 0
        def full_names(self): return ['Halley']
        def target_identifications(self):
            return [('Halley', ['1P'], 'Comet', ['A comet.'], 'halley')]
        def mpc_key(self): return '1P'
    monkeypatch.setattr('HST.target_identifications.comets.identify_comet', lambda keys, **kwargs: FakeComet())
    result = comet_identifications('1P')
    assert any('Halley' in tid[0] for tid in result)

# __init__.py (main entry points)
def test_hst_target_identifications_and_details(monkeypatch):
    # Patch dependencies to return simple results
    monkeypatch.setattr(ti, 'standard_body_identifications', lambda keys, include=[]: [('Earth', [], 'Planet', ['desc'], 'earth')])
    monkeypatch.setattr(ti, 'minor_planet_identifications', lambda keys, **kwargs: [('Chiron', [], 'Centaur', ['desc'], 'chiron')])
    monkeypatch.setattr(ti, 'comet_identifications', lambda keys, **kwargs: [('Halley', [], 'Comet', ['desc'], 'halley')])
    # Fake SPT header with TARG_ID and PROPOSID
    header = {'TARGNAME': 'Earth', 'TARG_ID': 'Earth', 'PROPOSID': '12345'}
    with pytest.raises(ValueError):
        ti.hst_target_identifications(header, filepath='fake.fits')
    # Details
    details = ti.target_identification_details(header)
    assert isinstance(details, list)
    # TNO survey and unknown TNO
    assert isinstance(ti.tno_survey(header), list)
    assert isinstance(ti.unknown_tno(header), list)
