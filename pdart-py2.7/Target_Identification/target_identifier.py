import os
import cspyce
import gravity
import oops
import oops.inst.hst as hst
import spicedb

import cspyce.aliases
cspyce.use_errors()
cspyce.use_aliases()

import roman
from nomenclature_exceptions import EXCEPTIONS, ALT_NAMES

oops.define_solar_system("1990-01-01", "2020-01-01")

LID_PREFIX = 'urn:nasa:pds:context:target:'

AU = 149597870.7
SSB_GM = gravity.SUN_JUPITER
SSB_ID = 0
SUN_ID = 10
PLANET_IDS = set(list(range(199,900,100)))
MOON_DIV_1000 = set(list(range(10,90,10)) + list(range(15,95,10)))

PLANET_NAMES = {
    199: 'Mercury',
    299: 'Venus',
    399: 'Earth',
    499: 'Mars',
    599: 'Jupiter',
    699: 'Saturn',
    799: 'Uranus',
    899: 'Neptune',
    999: '(134340) Pluto',
}

def get_spice_ids(spk_file):
    """Return the SPICE IDs from the SPK file."""

    spice_ids = cspyce.spkobj(spk_file)
    spice_ids.sort()
    return spice_ids

def _asteroid_centaur_tno(spice_id, spk_file):
    """Return the string "Asteroid", "Trans-Neptunian Object", or "Centaur"
    based on the SPICE ID and SPK file.
    semimajor axis.

    An asteroid has semimajor axis < 5.4 AU; a TNO has semimajor axis > 30.1;
    anything in between is a Centaur.

    The semimajor axis is calculated as an osculating orbital element at the
    earliest time found in the SPK file.
    """

    if spk_file is None:
        raise ValueError('SPICE ID %d cannot be categorized; ' % spice_id,
                         'missing SPK')

    first_times = cspyce.spkcov(spk_file, spice_id)[0]
    et = 0.5 * (first_times[0] + first_times[1])

    spicedb.furnish_by_filepath(spk_file, fast=True)
    (state, lt) = cspyce.spkez(spice_id, et, 'J2000', 'NONE', SSB_ID)
    osc = gravity.SUN_JUPITER.osc_from_state(state[:3], state[3:])

    a = osc[0] / AU
    if a < 5.4:
        return 'Asteroid'

    if a > 30.1:
        return 'Trans-Neptunian Object'

    return 'Centaur'

def get_target_type(spice_id, spk_file=None):
    """Return the target type based on the SPICE ID.

    See ftp://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/naif_ids.html
    """

    if spice_id >= 50000001:    # Fragments of SL9
        return 'Comet'

    if spice_id >= 2000001:
        if spk_file is None:
            return 'Minor Planet'
        return _asteroid_centaur_tno(spice_id, spk_file)

    if spice_id >= 1000001:
        return 'Comet'

    if spice_id in DWARF_PLANET_IDS:
        return 'Dwarf Planet'

    if spice_id in (99, 0):
        return 'Sun'

    if spice_id in PLANET_IDS:
        return 'Planet'

    if spice_id > 100 and spice_id < 999 and (spice_id % 100) != 99:
        return 'Satellite'

    if spice_id // 1000 in MOON_DIV_1000 and spice_id % 1000 != 0:
        return 'Satellite'

    raise ValueError('SPICE ID %d cannot be categorized' % spice_id)

def is_minor_planet(spice_id):
    """True if this the SPICE ID of a minor planet; False otherwise."""

    return (spice_id >= 2000001 and spice_id < 50000001)

def get_planet_spice_id(spice_id):
    """Return the SPICE ID of the body that this body orbits if it is not the
    Sun."""

    # Satellites of the planets and Pluto
    if spice_id > 100 and spice_id <= 999:
        return 100 * (spice_id // 100) + 99

    # Extended IDs used generally for preliminary identifications
    if spice_id // 1000 in MOON_DIV_1000 and spice_id % 1000 != 0:
        return 100 * (spice_id // 10000) + 99

    return spice_id

def get_satellite_number(spice_id):
    """Return the satellite number of this body, zero if it is a central body.
    """

    if spice_id > 100 and spice_id <= 999:
        satellite_number = spice_id % 100
        if satellite_number == 99:
            satellite_number = 0

        return satellite_number

    return 0

def get_lid(spice_id, target_name, target_type):
    """Return the LID for this target."""

    # Ignore case and underscores
    target_type = target_type.lower().replace(' ', '_')

    if target_type == 'satellite':
        planet_name = PLANET_NAMES[get_planet_spice_id(spice_id)]
        planet_name = planet_name.replace(' ', '_')
        target_name = planet_name + '.' + target_name
        target_name = target_name.lower().replace(' ', '')
    else:
        target_name = target_name.lower().replace(' ', '_')

    # Remove non-compliant characters
    body_list = list(target_name)
    for k in range(len(body_list)):
        c = body_list[k]
        if c not in ('abcdefghijklmnopqrstuvwxyz0123456789-_.'):
            body_list[k] = ''

    target_name = ''.join(body_list)

    # Special case Sun.Sun
    lid = LID_PREFIX + target_type
    if target_type != 'sun':
        lid = lid + '.' + target_name

    return lid

def get_solar_system_info(body_dict=None):
    """Return a dictionary of (spice_id, body names, body type[, primary ID])
    vs. SPICE ID for planets and moons of the solar system.

    If a dictionary is provided, the new bodies are added to it; otherwise, a
    new dictionary is returned."""

    if body_dict is None:
        body_dict = {}

    body_dict[SUN_ID] = (SUN_ID, ['Sun'], 'Sun')
    body_dict[199] = (199, ['Mercury'], 'Planet')
    body_dict[299] = (299, ['Venus'  ], 'Planet')
    body_dict[399] = (399, ['Earth'  ], 'Planet')
    body_dict[499] = (499, ['Mars'   ], 'Planet')
    body_dict[599] = (599, ['Jupiter'], 'Planet')
    body_dict[699] = (699, ['Saturn' ], 'Planet')
    body_dict[799] = (799, ['Uranus' ], 'Planet')
    body_dict[899] = (899, ['Neptune'], 'Planet')
    body_dict[999] = (999, ['(134340) Pluto', 'Pluto'], 'Dwarf Planet')

    planet_ids = list(range(199, 1000, 100))

    # Use the OOPS mechanism for keeping track of planetary satellites
    for planet_id in planet_ids:
        planet_names = body_dict[planet_id][1]
        short_name = planet_names[-1]

        moons = oops.Body.as_body(short_name).select_children(
                                                      include_all=['SATELLITE'])

        # For each moon...
        for body in moons:
            spice_id = body.spice_id

            # Check the exceptions list before proceeding
            if spice_id in EXCEPTIONS:
                body_dict[spice_id] = EXCEPTIONS[spice_id]
                continue

            name = body.name

            # A provisional name always contains an underscore; a formally
            # approved name does not
            if '_' in name:

                # Convert a provisional name to standard notation
                names = ['S/' + name[1:].replace('_', ' ')]
                name_is_provisional = True

            else:
                # Repair case of a formal name
                names = [name[0].upper() + name[1:].lower()]
                name_is_provisional = False

            # Satellite numbers are encoded in the SPICE ID
            number = get_satellite_number(spice_id)
            for planet_name in planet_names:    # Because Pluto has two names
              if number:
                if name_is_provisional:
                    names.append('%s %s' % (planet_name,
                                            roman.int_to_roman(number)))
                else:
                    names.append('%s %s (%s)' % (planet_name,
                                                 roman.int_to_roman(number),
                                                 names[0]))

            # We track alternative names, e.g., provisional names of newly-named
            # moons, manually
            if name in ALT_NAMES:
                names += ALT_NAMES[name]

            # Update the dictionary
            body_dict[spice_id] = (spice_id, names, 'Satellite', planet_id)

    return body_dict

def target_identification(spice_id, body_dict):
    """Return the PDS4-compliant Target_Identification Object" as a list of
    strings."""

    info = body_dict[spice_id]
    (body_names, target_type) = info[1:3]
    target_name = body_names[0]

    if len(info) > 3:
        primary_id = info[3]
    else:
        primary_id = ''

    records = []
    records.append('<Target_Identification>')
    records.append('  <name>')
    records.append('    %s' % target_name)
    records.append('  </name>')

    for name in body_names[1:]:
        records.append('  <alternate_designation>')
        records.append('    %s' % name)
        records.append('  </alternate_designation>')

    records.append('  <alternate_designation>')
    records.append('    NAIF ID %s' % spice_id)
    records.append('  </alternate_designation>')

    records.append('  <type>')
    records.append('    %s' % target_type)
    records.append('  </type>')

    if primary_id:
        primary_info = body_dict[primary_id]
        primary_name = primary_info[1][0]
        primary_type = primary_info[2]

        records.append('  <description>')
        records.append('    Satellite of: %s;' % primary_name)
        records.append('    Type of primary: %s;' % primary_type)
        records.append('    LID of primary: %s;' % get_lid(primary_id,
                                                          primary_name,
                                                          primary_type))
        records.append('    NAIF ID of primary: %s;' % primary_id)
        records.append('  </description>')

    records.append('  <Internal_Reference>')
    records.append('    <lid_reference>')
    records.append('      %s' % get_lid(spice_id, target_name, target_type))
    records.append('    </lid_reference>')
    records.append('  </Internal_Reference>')

    records.append('</Target_Identification>')
    records.append('')

    return records

################################################################################
