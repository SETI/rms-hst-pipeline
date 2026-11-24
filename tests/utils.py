import os
import shutil

from hst_pipeline.citations import Citation_Information
from hst_pipeline.product_labels.suffix_info import INSTRUMENT_NAMES

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
    'instrument_name_dict': INSTRUMENT_NAMES,
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
    return os.path.join(os.path.dirname(__file__), basename)

def golden_file_contents(basename):
    """
    Return the contents as a list of strings of the golden file with
    the given basename.
    """
    with open(golden_filepath(basename), "r") as f:
        return f.readlines()

# Lines whose content is provenance rather than data: they change from run to run or
# machine to machine, so a golden file can never pin them down.
#   modification_date  the date the label was written.
#   <!--               the header comments, which carry the label version and date.
#   pipeline version   embeds LABEL_DATE, which product_labels derives from the mtime of
#                      its own source file, so it changes whenever that file is touched.
#   software version   embeds the version of whichever rms-pdstemplate is installed.
_VOLATILE_MARKERS = ('modification_date', '<!--', 'pipeline version', 'software version')


def _is_volatile(line):
    """Whether this line holds provenance that a golden file cannot pin down."""
    return any(marker in line for marker in _VOLATILE_MARKERS)


def assert_golden_file_equal(basename, calculated_contents):
    """Compare generated label contents against the golden copy.

    Lines holding provenance (see _VOLATILE_MARKERS) are skipped; every other line must
    match exactly, and the two files must have the same number of lines. If the golden
    file does not exist it is written and the test fails, so that the new file is
    reviewed and committed deliberately.

    Inputs:
        basename             path of the golden file, relative to the tests directory.
        calculated_contents  the generated label, as a list of lines.
    """
    filepath = golden_filepath(basename)

    if not os.path.isfile(filepath):
        with open(filepath, 'w') as f:
            for line in calculated_contents:
                f.write(str(line))
        assert False, f"Golden file {basename!r} did not exist but it was created."

    with open(filepath, 'r') as f:
        expected = f.readlines()

    assert len(expected) == len(calculated_contents), (
        f'{basename}: golden file has {len(expected)} lines, generated label has '
        f'{len(calculated_contents)}')

    for lineno, (want, got) in enumerate(zip(expected, calculated_contents), start=1):
        if _is_volatile(want) or _is_volatile(got):
            continue
        assert want == got, (f'{basename} line {lineno}:\n'
                             f'  golden   : {want.rstrip()!r}\n'
                             f'  generated: {got.rstrip()!r}')

def remove_dirs(directories):
    for path in directories:
         shutil.rmtree(path, ignore_errors=True)

# A miniature HST handbooks collection: each label's name, LID (None for a collection
# label), Citation_Information keywords, and the documents it describes
_LID = 'urn:nasa:pds:hst_handbooks:document:'
HANDBOOK_LABELS = [
    ('acs-data-handbook.lblx', _LID + 'acs-data-handbook', ['HST', 'ACS'],
     ['ACS-Data-Handbook-2024-v13.pdf']),
    ('fos-data-handbook.lblx', _LID + 'fos-data-handbook', ['HST', 'FOS'],
     ['FOS-Data-Handbook-1998-v3.1-final.pdf', 'FOS-Data-Handbook-correction-2001.pdf']),
    ('hst-data-handbook-volume-1.lblx', _LID + 'hst-data-handbook-volume-1',
     ['HST', 'FOC'], ['HST-Data-Handbook-Volume-1-1997-v3-OCR.pdf']),
    ('hst-data-handbook-volume-2.lblx', _LID + 'hst-data-handbook-volume-2',
     ['HST', 'FOS', 'GHRS', 'HSP', 'WFPC2'], ['HST-Data-Handbook-Volume-2-1997-v3-OCR.pdf']),
    ('nicmos-data-handbook.lblx', _LID + 'nicmos-data-handbook', ['HST', 'NICMOS'],
     ['NICMOS-Data-Handbook-2009-v8.pdf']),
    ('nicmos-instrument-handbook.lblx', _LID + 'nicmos-instrument-handbook',
     ['HST', 'NICMOS'], ['NICMOS-Instrument-Handbook-2009-v11-final.pdf']),
    ('wfpc-data-handbook.lblx', _LID + 'wfpc-data-handbook', ['HST', 'wfpc'],
     ['WFPC-Data-Handbook-final.pdf']),
    ('wfpc2-data-handbook.lblx', _LID + 'wfpc2-data-handbook', ['HST', 'WFPC2'],
     ['WFPC2-Data-Handbook-2010-v5-final.pdf']),
    ('collection_document.lblx', None, ['HST', 'ACS', 'NICMOS'],
     ['collection_document.csv']),
]

def handbook_label_text(lid, keywords, files):
    """The text of a minimal handbook label, in the PDS4 namespace; a collection label
    if lid is None."""
    root = 'Product_Collection' if lid is None else 'Product_Document'
    lid = _LID.rstrip(':') if lid is None else lid
    keywords = ''.join(f'            <keyword>{k}</keyword>\n' for k in keywords)
    files = ''.join(f'            <Document_File><file_name>{f}</file_name>'
                    '</Document_File>\n' for f in files)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<{root} xmlns="http://pds.nasa.gov/pds4/pds/v1">\n'
            '    <Identification_Area>\n'
            f'        <logical_identifier>{lid}</logical_identifier> <!--TBD-->\n'
            '        <Citation_Information>\n'
            f'{keywords}'
            '        </Citation_Information>\n'
            '    </Identification_Area>\n'
            '    <Document>\n'
            f'{files}'
            '    </Document>\n'
            f'</{root}>\n')

def make_handbooks_dir(path):
    """Write a miniature handbooks collection (HANDBOOK_LABELS and the documents they
    describe) into the directory at path, creating it if necessary. Return the path."""
    os.makedirs(path, exist_ok=True)
    for (name, lid, keywords, files) in HANDBOOK_LABELS:
        with open(os.path.join(path, name), 'w') as f:
            f.write(handbook_label_text(lid, keywords, files))
        for file in files:
            with open(os.path.join(path, file), 'w') as f:
                f.write(f'contents of {file}\n')
    return str(path)
