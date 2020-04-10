import os
import subprocess
import cspyce
import spicedb
import gravity

from nomenclature_exceptions import EXCEPTIONS
from target_identifier import *

AU = 149597870.7    # km
SSB_GM = gravity.SUN_JUPITER
SSB_ID = 0
SUN_ID = 10
PLANET_IDS = set(list(range(199,900,100)))
MOON_DIV_1000 = set(list(range(10,90,10)) + list(range(15,95,10)))

DUPLICATED_NAMES = set([
    'Metis', 'Adrastea', 'Amalthea', 'Io', 'Europa', 'Leda',
    'Pan', 'Prometheus', 'Pandora', 'Epimetheus', 'Dione', 'Rhea',
    'Cordelia', 'Ophelia', 'Bianca', 'Desdemona', 'Titania',
    'Galatea', 'Larissa',
    'Halley',
])

def _read_comments(spk_file):
    """Read the comments from an SPK file into a list of strings."""

    cmt_file = spk_file[:-4] + '.cmt'

    if not os.path.exists(cmt_file):
        subprocess.call(['commnt', '-e', spk_file, cmt_file])

    if not os.path.exists(cmt_file):
        raise IOError('unable to read comments from ' +
                      os.path.basename(spk_file))

    with open(cmt_file, 'r') as f:
        recs = f.readlines()

    return recs

def get_body_info(spk_file):
    """Return (SPICE ID, body names, target type) from a Horizons SPK file."""

    spice_id = None
    body_name = ''

    # Read info from the Horizons comment section
    try:
        recs = _read_comments(spk_file)
        for rec in recs:
            if rec.startswith('Target body'):
                parts = rec.split(':')
                body_name = parts[1].split('{')[0]
            if rec.startswith('Target SPK ID'):
                parts = rec.split(':')
                spice_id = int(parts[1].strip())

    # Otherwise, try scraping info from the SPK filename
    except IOError:
        (spice_id, body_name) = _get_body_info_from_filename(spk_file)

    if spice_id is None:
        raise IOError('SPICE ID not found in ' + basename)

    # Return info from the exceptions list if found
    if spice_id in EXCEPTIONS:
        return (spice_id,) + EXCEPTIONS[spice_id]

    # Split the body name string if possible
    parts = body_name.partition('(')
    if parts[2]:
        body_name1 = parts[0].strip()
        body_name2 = parts[2].strip().strip(')').strip()
    else:
        body_name1 = parts[0].strip()
        body_name2 = ''

    # Determine the target type
    target_type = get_target_type(spice_id)
    is_minor_planet = (target_type == 'Minor Planet')
    if is_minor_planet:
        target_type = get_target_type(spice_id, spk_file)

    # Construct the primary designation
    if is_minor_planet:
        parts = body_name1.split(' ')
        try:
            number = int(parts[0])
            name = ' '.join(parts[1:])
            body_names = ['(%d) %s' % (number, name)]

            if name not in DUPLICATED_NAMES:
                body_names.append(name)

            if body_name2:
                body_names.append(body_name2)

            body_names.append('Minor Planet %d' % number)

        except ValueError:
            pass

    else:
        if body_name2:
            body_names = ['%s %s' % (body_name2, body_name1)]
            if body_name1 not in EXCEPTIONS:
                body_names.append(body_name1)
        else:
            body_names = [body_name1]

    return (spice_id, body_names, target_type)

def _get_body_info_from_filename(spk_file):
    """Return the SPICE ID and body name from the file name.

    This version recognizes the standard file naming procedure used by the
    PDS Ring-Moon Systems and Small Bodies Node.
    """

    # Example: ID2060558_Echeclus_19900101_20301231_20171207.bsp

    basename = os.path.basename(spk_file)
    if basename[:2].upper() != 'ID':
        raise ValueError('Not a recognized SPK filename format: ' + basename)

    parts = basename.split('_')
    spice_id = int(parts[0][2:])
    parts = parts[1:]

    # Determine the target type
    target_type = get_target_type(spice_id)
    is_minor_planet = (target_type == 'Minor Planet')

    # Remove dates
    for k in range(3):
        if parts[-1][:2] in ('18', '19', '20', '21'):
            parts = parts[:-1]

    # Assemble the body name
    if target_type == 'Minor Planet':
        number = str(spice_id - 2000000)
        if number in parts[0]:
            parts = parts[1:]
        body_name = number + ' ' + ' '.join(parts)

    elif targ_type == 'Comet':
        if len(parts[-3]) == 1 and parts[-3] in 'PCXDAI':
            body_name = ' '.join(parts[:-3]) + ' (%s/%s %s)' % tuple(parts[-3:])

    else:
        body_name = ' '.join(parts)

    return (spice_id, body_name)

def load_bodies(dir, body_dict=None):
    """Load Body objects for all of the SPKs in this directory tree. Also return
    a dictionary of (spice id, body names, body type) vs. SPICE ID."""

    if body_dict is None:
        body_dict = {}

    for (root, dirs, files) in os.walk(dir, followlinks=True):
      for file in files:
        if not file.lower().endswith('.bsp'): continue

        filepath = os.path.join(root, file)

        # Load the kernel
        spicedb.furnish_by_filepath(filepath, fast=True)

        # Save the target info
        (spice_id, body_names, target_type) = get_body_info(filepath)
        body_dict[spice_id] = (spice_id, body_names, target_type)

        # Define within OOPS
        oops.define_small_body(spice_id, body_names[0], filepath)

    return body_dict
