from fs.path import dirname, join
import os

from citations import Citation_Information

TEST_COPIES_DIR = 'golden_copies'

LBL_DATA_DICT = {
    'prop_id': '7885',
    'collection_name': 'schema',
    'citation_info': Citation_Information(
        '',
        '7885',
        'GO',
        7,
        ['Heidi Hammel', 'Erich Karkoschka', 'Mark Marley'],
        'Uranus Nearing Equinox: Vertical Aerosol Distribution ' +
        'of Atmospheric Structure',
        1998,
        0,
        ['Recent HST images confirm that Uranus is undergoing atmospheric ' +
            'changes as the planet inches toward equinox in 2007 (parts of its ' +
            'northern hemisphere are now moving into sunlight after being in ' +
            'darkness for many years). GO 6818 (Hammel et al.) yielded excellent ' +
            'PC1 imaging of Uranus (NICMOS provides even better discrimination ' +
            'for vertical aerosol distribution than PC1, but was not available ' +
            'in Cycle 6). The data revealed strongly wavelength-dependent ' +
            'latitudinal structure and the presence of visible-wavelength ' +
            'cloud features in the northern hemisphere - the first ever so ' +
            'detected in the modern era. One orbit of GO 7429 (Tomasko and ' +
            'Karkoschka) was devoted to Uranus NICMOS imaging in six filters. ' +
            'Like the PC1 images, the NICMOS images show strongly ' +
            'wavelength-dependent latitudinal banding and reveal several ' +
            'discrete features near 30 degrees north - these features have ' +
            'the highest contrast ever seen for a Uranian cloud. The ' +
            'timescale of the changes on Uranus is unknown, but some models ' +
            'predict it could be rapid. Furthermore, this is the first ' +
            'opportunity to detect some regions of the northern hemisphere ' +
            'on Uranus. NICMOS will be used to image Uranus during this ' +
            'unique epoch of change. Adequate sampling of all longitudes in ' +
            'the northern hemisphere, where the newly detected features ' +
            'appear, requires 3 orbits. When possible, observations will be ' +
            'coordinated with ground-based imaging, spectroscopy, and photometry.']
    ),
    'formatted_title': 'Uranus Nearing Equinox: Vertical Aerosol Distribution ' +
                        'of Atmospheric Structure, HST Cycle 7 Program 7885, ' +
                        '1999.',
    'processing_level': 'Raw',
    'wavelength_ranges': ['Near Infrared', 'Infrared'],
    'instrument_name_dict': {'ACS': 'Advanced Camera for Surveys',
                                'COS': 'Cosmic Origins Spectrograph',
                                'FGS': 'Fine Guidance Sensors',
                                'FOC': 'Faint Object Camera',
                                'FOS': 'Faint Object Spectrograph',
                                'GHRS': 'Goddard High Resolution Spectrograph',
                                'HSP': 'High Speed Photometer',
                                'NICMOS': 'Near-Infrared Camera and' +
                                        ' Multi-Object Spectrometer',
                                'STIS': 'Space Telescope Imaging Spectrograph',
                                'WFC3': 'Wide Field Camera 3',
                                'WFPC': 'Wide Field/Planetary Camera',
                                'WFPC2': 'Wide Field and Planetary Camera 2'},
    'target_identifications': [
        {'name': 'Uranus',
            'alternate_designations': [],
            'type': 'Planet',
            'description': '\n        NAIF ID: 799\n      ',
            'formatted_name': 'uranus',
            'formatted_type': 'planet',
            'lid': 'urn:nasa:pds:context:target:planet.uranus'}
    ],
    'label_date': '2023-03-17',
    'inst_id_li': ['NICMOS'],
    'start_date_time': '1998-08-05T01:26:05Z',
    'stop_date_time': '1998-08-08T23:12:12Z',
    'start_date': '1998-08-05',
    'stop_date': '1998-08-08',
    'version_id': (1, 0),
    'csv_filename': 'collection_schema.csv',
    'records_num': 3,
    'mod_history': []
}

def golden_filepath(basename):
    """Return the path to a golden file with the given basename."""
    return join(dirname(__file__), basename)

def golden_file_contents(basename):
    """
    Return the contents as a list of strings of the golden file with
    the given basename.
    """
    with open(golden_filepath(basename), "r") as f:
        return f.readlines()

def assert_golden_file_equal(basename, calculated_contents):
    """
    Finds the golden file and compares its contents with the given string.
    Raises an exception via the unittest.TestCase argument if they are
    unequal. If the file does not yet exist, it writes the given string to
    the file. Inventories must be written with CR/NL line-termination, so
    a parameter for that is provided.
    """
    filepath = golden_filepath(basename)
    if os.path.isfile(filepath):
        contents = golden_file_contents(filepath)
        for idx, line in enumerate(contents):
            # bypass the current date and comments
            if 'modification_date' or '<!--' in line:
                continue
            assert line == calculated_contents[idx]
    else:
        with open(filepath, "w") as f:
            for line in calculated_contents:
                f.write(str(line))
            assert False, f"Golden file {basename!r} did not exist but it was created."
