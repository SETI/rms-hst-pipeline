##########################################################################################
# tests/test_label_creations_with_data_dict.py
#
# Tests related to label creations using preprogrammed data dictionary.
##########################################################################################

import os
import pytest
import shutil
import tempfile

from .utils import (assert_golden_file_equal,
                    golden_file_contents,
                    handbook_label_text,
                    make_handbooks_dir,
                    TEST_COPIES_DIR,
                    LBL_DATA_DICT)
from hst_pipeline.finalize_context import finalize_hst_context_directory
from hst_pipeline.finalize_data_product import COL_DATA_LABEL_TEMPLATE
from hst_pipeline.finalize_document import (copy_handbook_files,
                                            copy_mast_proposal_page,
                                            expected_handbook_lids,
                                            finalize_hst_document_directory,
                                            get_document_files,
                                            get_document_sources,
                                            get_handbook_files,
                                            get_handbook_instrument,
                                            get_handbooks_dir,
                                            HandbooksError,
                                            strip_mast_proposal_page)
from hst_pipeline.finalize_schema import finalize_hst_schema_directory
from hst_pipeline.hst_helper.fs_utils import (create_col_dir_in_bundle,
                                              get_deliverable_path,
                                              get_program_dir_path)
from hst_pipeline.hst_helper.general_utils import create_collection_label
from hst_pipeline.label_bundle import label_hst_bundle

class TestLabelCreationsWithDataDict:
    def setup_method(self):
        # data dictionary used to create the label
        self.data_dict = LBL_DATA_DICT

        # Make the temporary directories for testing
        self.testing_dir = [get_program_dir_path('7885', None, 'bundles', True)]
        for temp_dir in self.testing_dir:
            os.mkdir(temp_dir)

        # A miniature handbooks collection for the document collection
        self.handbooks_dir = make_handbooks_dir(tempfile.mkdtemp())

    def teardown_method(self):
        # Remove the testing directories
        for testing_dir in self.testing_dir:
            shutil.rmtree(testing_dir)
        shutil.rmtree(self.handbooks_dir)

    # Test schema colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_finalize_hst_schema_directory(self, p_id):
        sch_col_lbl = finalize_hst_schema_directory(p_id, self.data_dict, None, True)

        if os.path.isfile(sch_col_lbl):
            calculated_contents = golden_file_contents(sch_col_lbl)
        assert_golden_file_equal(f'{TEST_COPIES_DIR}/test_schema_col_label.golden.lblx',
                                 calculated_contents)

    # Test context colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_finalize_hst_context_directory_ctxt_col_lbl(self, p_id):
        data_dict = {
            'collection_name': 'context',
            'csv_filename': 'collection_context.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        ctxt_col_lbl, _ = finalize_hst_context_directory(p_id, data_dict, None, True)

        if os.path.isfile(ctxt_col_lbl):
            calculated_contents = golden_file_contents(ctxt_col_lbl)
        assert_golden_file_equal(f'{TEST_COPIES_DIR}/test_context_col_label.golden.lblx',
                                 calculated_contents)

    # Test that the context collection lists each target once, by its LID
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_finalize_hst_context_directory_target_lids(self, p_id):
        prefix = 'urn:nasa:pds:context:target:'
        fragment = {'name': '73P/Schwassmann-Wachmann 3 B', 'type': 'Comet',
                    'alternate_designations': ['73P/Schwassmann-Wachmann 3-B'],
                    'lid': prefix + 'comet.73p_schwassmann-wachmann_3-b'}
        variant = {**fragment, 'alternate_designations': []}
        moon = {'name': 'Io', 'type': 'Satellite', 'alternate_designations': [],
                'lid': prefix + 'satellite.jupiter.io'}
        data_dict = {**self.data_dict,
                     'target_identifications': [fragment, moon, variant]}
        finalize_hst_context_directory(p_id, data_dict, None, True)

        _, context_dir = create_col_dir_in_bundle(p_id, 'context', True)
        with open(f'{context_dir}/collection_context.csv') as f:
            targets = [line.strip() for line in f if ':target:' in line]
        assert targets == [f'S,{prefix}comet.73p_schwassmann-wachmann_3-b::1.0',
                           f'S,{prefix}satellite.jupiter.io::1.0']

        with open(f'{context_dir}/collection_context.lblx') as f:
            assert '<records>5</records>' in f.read()

    # Test investigation label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_finalize_hst_context_directory_inv_lbl(self, p_id):
        data_dict = {
            'collection_name': 'context',
            'csv_filename': 'collection_context.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        _, inv_lbl = finalize_hst_context_directory(p_id, data_dict, None, True)

        if os.path.isfile(inv_lbl):
            calculated_contents = golden_file_contents(inv_lbl)
        assert_golden_file_equal(f'{TEST_COPIES_DIR}/test_investigation_label.golden.lblx',
                                 calculated_contents)

    # Test document colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_finalize_hst_document_directory_doc_col_lbl(self, p_id):
        data_dict = {
            'collection_name': 'document',
            'csv_filename': 'collection_document.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        doc_col_lbl, _ = finalize_hst_document_directory(p_id, data_dict, None, True,
                                                         self.handbooks_dir)

        if os.path.isfile(doc_col_lbl):
            calculated_contents = golden_file_contents(doc_col_lbl)
        assert_golden_file_equal(f'{TEST_COPIES_DIR}/test_document_col_label.golden.lblx',
                                 calculated_contents)

        # The NICMOS handbooks are secondary members, by LID, and are copied with their
        # labels into the collection
        document_dir = os.path.dirname(doc_col_lbl)
        with open(f'{document_dir}/collection.csv', newline='') as f:
            assert f.read().splitlines() == [
                'P,urn:nasa:pds:hst_07885:document:07885::1.0',
                'S,urn:nasa:pds:hst_handbooks:document:nicmos-data-handbook',
                'S,urn:nasa:pds:hst_handbooks:document:nicmos-instrument-handbook',
            ]
        assert sorted(f for f in os.listdir(document_dir) if f[0].lower() == 'n') == [
            'NICMOS-Data-Handbook-2009-v8.pdf',
            'NICMOS-Instrument-Handbook-2009-v11-final.pdf',
            'nicmos-data-handbook.lblx',
            'nicmos-instrument-handbook.lblx',
        ]

    # Test document label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_finalize_hst_document_directory_doc_lbl(self, p_id):
        data_dict = {
            'collection_name': 'document',
            'csv_filename': 'collection_document.csv',
        }
        data_dict = {**self.data_dict, **data_dict}
        _, doc_lbl = finalize_hst_document_directory(p_id, data_dict, None, True,
                                                     self.handbooks_dir)

        if os.path.isfile(doc_lbl):
            calculated_contents = golden_file_contents(doc_lbl)
        assert_golden_file_equal(f'{TEST_COPIES_DIR}/test_document_label.golden.lblx',
                                 calculated_contents)

    # Test data product colleciton label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_create_collection_label_data_prod_col_lbl(self, p_id):
        data_dict = {
            'collection_name': 'data_nicmos_cal',
            'csv_filename': 'collection_document.csv',
            'processing_level': 'Calibrated',
            'inst_id': 'nicmos',
            'collection_title': 'Calibrated NICMOS "_cal" image files from HST ' +
                                'Program 7885.',
            'instrument_name': 'HST Near Infrared Camera and Multi-Object Spectrometer',
            'stop_date_time': '1998-08-05T03:02:11Z',
        }
        data_dict = {**self.data_dict, **data_dict}

        col_name = data_dict['collection_name']
        # Create data product collection directory
        create_col_dir_in_bundle(p_id, col_name, True)
        col_data_label_name = f'collection_{col_name}.lblx'
        data_prod_col_lbl = create_collection_label(p_id, col_name,
                                                    data_dict, col_data_label_name,
                                                    COL_DATA_LABEL_TEMPLATE, None, True)

        if os.path.isfile(data_prod_col_lbl):
            calculated_contents = golden_file_contents(data_prod_col_lbl)
        assert_golden_file_equal(f'{TEST_COPIES_DIR}/test_data_prod_col_label.golden.lblx',
                                 calculated_contents)

    # Test bundle label creation
    @pytest.mark.parametrize('p_id', [('7885')])
    def test_label_hst_bundle(self, p_id):
        data_dict = {
            'collection_name': 'bundle',
            'bundle_entry_li': [
                ('context', 'context', (1, 0)),
                ('miscellaneous_nicmos_spt', 'miscellaneous', (1, 0)),
                ('miscellaneous_nicmos_trl', 'miscellaneous', (1, 0)),
                ('miscellaneous_nicmos_jif', 'miscellaneous', (1, 0)),
                ('document', 'document', (1, 0)),
                ('miscellaneous_nicmos_jit', 'miscellaneous', (1, 0)),
                ('schema', 'schema', (1, 0)),
                ('browse_nicmos_cal', 'browse', (1, 0)),
                ('data_nicmos_ima', 'data', (1, 0)),
                ('browse_nicmos_raw', 'browse', (1, 0))
            ]
        }
        data_dict = {**self.data_dict, **data_dict}

        deliverable_path = get_deliverable_path(p_id, True)
        os.makedirs(deliverable_path, exist_ok=True)
        bundle_lbl = label_hst_bundle(p_id, data_dict, None, True)

        if os.path.isfile(bundle_lbl):
            calculated_contents = golden_file_contents(bundle_lbl)
        assert_golden_file_equal(f'{TEST_COPIES_DIR}/test_bundle_label.golden.lblx',
                                 calculated_contents)


@pytest.mark.parametrize('files,expected', [
    # Proposal files are delivered; program-info.txt and unrelated files are not
    (['07885.pro', '07885.prop', '07885.pdf', 'program-info.txt', 'products.txt'],
     [('07885', '07885.pdf', 'PDF'),
      ('07885', '07885.pro', '7-Bit ASCII Text'),
      ('07885', '07885.prop', '7-Bit ASCII Text')]),
    # With an .apt file, a stale MAST page is not delivered
    (['07885.apt', '07885.html'], [('07885', '07885.apt', 'UTF-8 Text')]),
    # Without an .apt or .pro file, the MAST proposal page is delivered
    (['07885.html', 'program-info.txt'], [('07885', '07885.html', 'HTML')]),
])
def test_get_document_files(tmp_path, files, expected):
    for name in files:
        (tmp_path / name).write_text('x')
    assert get_document_files(7885, str(tmp_path)) == expected


MAST_PAGE = """<html lang="en"><head><title>HST Proposal Search</title></head><body>
<div id="nav"><a href="/">MAST</a></div>
<div id="indt">
<h2 align='center'>DO NEPTUNE AND PLUTO HAVE RINGS?<br> HST Proposal 1086</h2>
<center><H3>Robert Bless<br>University of Wisconsin - Madison</H3></center>
   <HR><H2>Proposal Abstract</H2>
   <p> The origin of planetary ring systems remains unknown. Caf\xe9.</p><hr/>
<h2>Publications referencing this proposal</h2>
<br><ul><li>No papers listed in database for this Proposal ID</ul><br><hr>
</div>
<h2> Data for Program ID: 1086 (as of now)</h2>
<div id="footernav">Top of Page</div>
</body></html>
"""


def test_strip_mast_proposal_page():
    stripped = strip_mast_proposal_page(MAST_PAGE)
    # Begins with a rule, then the title; ends after the publications
    assert stripped.startswith("<hr>\n<h2 align='center'>DO NEPTUNE AND PLUTO")
    assert stripped.endswith('No papers listed in database for this Proposal ID</ul>'
                             '<br><hr>\n')
    for dropped in ('<html', '<head>', 'id="nav"', 'Data for Program ID', 'footernav'):
        assert dropped not in stripped
    assert strip_mast_proposal_page('<html>No proposal found</html>') is None


def test_copy_mast_proposal_page(tmp_path):
    src = tmp_path / 'src.html'
    src.write_text(MAST_PAGE, encoding='latin-1')
    dst = tmp_path / 'dst.html'
    copy_mast_proposal_page(str(src), str(dst))
    text = dst.read_text(encoding='latin-1')
    assert text.startswith('<hr>\n<h2')
    assert 'Caf\xe9.' in text          # Latin-1 characters survive

    # A page without a proposal title is copied unchanged
    src.write_text('<html>No proposal found</html>', encoding='latin-1')
    copy_mast_proposal_page(str(src), str(dst))
    assert dst.read_text(encoding='latin-1') == '<html>No proposal found</html>'


def test_get_document_sources(tmp_path):
    # Only the MAST page has a source URL; its date is the saved page's modification date
    page = tmp_path / '01086.html'
    page.write_text('x')
    os.utime(page, (0, 1_000_000_000))      # 2001-09-09 UTC
    (tmp_path / '01086.pro').write_text('x')
    doc_files = [('01086', '01086.html', 'HTML'), ('01086', '01086.pro', '7-Bit ASCII Text')]
    sources = get_document_sources(1086, str(tmp_path), doc_files)
    assert len(sources) == 1
    (file, url, date) = sources[0]
    assert file == '01086.html'
    assert url == 'https://archive.stsci.edu/proposal_search.php?id=1086&mission=hst'
    assert date in ('2001-09-08', '2001-09-09')     # depends on the local time zone
    assert get_document_sources(5825, str(tmp_path), doc_files[1:]) == []


@pytest.mark.parametrize('filename,expected', [
    ('WFPC2-Data-Handbook-2010-v5-final.pdf', 'WFPC2'),
    ('wfpc2-data-handbook.lblx',              'WFPC2'),
    ('hst-data-handbook-volume-1.lblx',       'HST'),
    ('FOS-Data-Handbook-correction-2001.pdf', 'FOS'),
    ('collection_document.csv',               None),
    ('readme.txt',                            None),
])
def test_get_handbook_instrument(filename, expected):
    assert get_handbook_instrument(filename) == expected


def test_get_handbook_files(tmp_path):
    handbooks_dir = make_handbooks_dir(tmp_path)

    # Every handbook whose label has the instrument as a keyword: the label, the
    # documents it names, and its LID
    (lids, files) = get_handbook_files(['NICMOS'], handbooks_dir)
    assert lids == ['urn:nasa:pds:hst_handbooks:document:nicmos-data-handbook',
                    'urn:nasa:pds:hst_handbooks:document:nicmos-instrument-handbook']
    assert files == ['NICMOS-Data-Handbook-2009-v8.pdf',
                     'NICMOS-Instrument-Handbook-2009-v11-final.pdf',
                     'nicmos-data-handbook.lblx',
                     'nicmos-instrument-handbook.lblx']

    # One label can describe two documents; keywords are matched ignoring case; WFPC
    # does not pick up WFPC2's handbooks; a volume of the HST Data Handbook goes with
    # each instrument it names
    (lids, files) = get_handbook_files({'FOS', 'wfpc'}, handbooks_dir)
    assert lids == ['urn:nasa:pds:hst_handbooks:document:fos-data-handbook',
                    'urn:nasa:pds:hst_handbooks:document:hst-data-handbook-volume-2',
                    'urn:nasa:pds:hst_handbooks:document:wfpc-data-handbook']
    assert files == ['FOS-Data-Handbook-1998-v3.1-final.pdf',
                     'FOS-Data-Handbook-correction-2001.pdf',
                     'HST-Data-Handbook-Volume-2-1997-v3-OCR.pdf',
                     'WFPC-Data-Handbook-final.pdf',
                     'fos-data-handbook.lblx',
                     'hst-data-handbook-volume-2.lblx',
                     'wfpc-data-handbook.lblx']

    # A file name no longer decides: the FOC has only Volume I, GHRS only Volume II
    assert get_handbook_files(['FOC'], handbooks_dir) == (
        ['urn:nasa:pds:hst_handbooks:document:hst-data-handbook-volume-1'],
        ['HST-Data-Handbook-Volume-1-1997-v3-OCR.pdf', 'hst-data-handbook-volume-1.lblx'])
    assert get_handbook_files(['GHRS'], handbooks_dir)[0] == [
        'urn:nasa:pds:hst_handbooks:document:hst-data-handbook-volume-2']

    # The collection label's keywords don't count, nor does the HST keyword match an
    # instrument; an instrument without handbooks contributes nothing
    assert get_handbook_files(['ACS'], handbooks_dir)[1] == [
        'ACS-Data-Handbook-2024-v13.pdf', 'acs-data-handbook.lblx']
    assert get_handbook_files(['HSP', 'STIS'], handbooks_dir)[0] == [
        'urn:nasa:pds:hst_handbooks:document:hst-data-handbook-volume-2']
    assert get_handbook_files(['STIS'], handbooks_dir) == ([], [])

    # A label without a LID, one that isn't XML, and one naming a missing document are
    # errors
    label = tmp_path / 'stis-data-handbook.lblx'
    label.write_text(handbook_label_text('', ['STIS'], []))
    with pytest.raises(HandbooksError, match='No logical_identifier.*stis-data-handbook'):
        get_handbook_files(['ACS'], handbooks_dir)
    label.write_text('<Product_Document>')
    with pytest.raises(HandbooksError, match='Cannot parse.*stis-data-handbook'):
        get_handbook_files(['ACS'], handbooks_dir)
    label.write_text(handbook_label_text('urn:x', ['STIS'], ['STIS-missing.pdf']))
    with pytest.raises(HandbooksError, match='missing file: STIS-missing.pdf'):
        get_handbook_files(['STIS'], handbooks_dir)
    assert get_handbook_files(['ACS'], handbooks_dir)[0] == [
        'urn:nasa:pds:hst_handbooks:document:acs-data-handbook']


def test_expected_handbook_lids():
    assert expected_handbook_lids(['WFPC', 'acs']) == [
        'urn:nasa:pds:hst_handbooks:document:wfpc-data-handbook',
        'urn:nasa:pds:hst_handbooks:document:wfpc-instrument-handbook',
        'urn:nasa:pds:hst_handbooks:document:acs-data-handbook',
        'urn:nasa:pds:hst_handbooks:document:acs-instrument-handbook',
    ]


def test_copy_handbook_files(tmp_path):
    handbooks_dir = make_handbooks_dir(tmp_path / 'handbooks')
    document_dir = tmp_path / 'document'
    document_dir.mkdir()
    (document_dir / '07885.pro').write_text('proposal')

    (_, files) = get_handbook_files(['NICMOS', 'ACS'], handbooks_dir)
    copy_handbook_files(files, handbooks_dir, str(document_dir))
    assert sorted(os.listdir(document_dir)) == sorted(files + ['07885.pro'])

    # An unchanged copy is left alone; a changed handbook is copied again
    copy = document_dir / 'nicmos-data-handbook.lblx'
    original = os.stat(f'{handbooks_dir}/nicmos-data-handbook.lblx')
    copy.write_text('x' * original.st_size)
    os.utime(copy, (0, original.st_mtime))
    copy_handbook_files(files, handbooks_dir, str(document_dir))
    assert copy.read_text().startswith('xxx')   # same size and time: not recopied
    with open(f'{handbooks_dir}/NICMOS-Data-Handbook-2009-v8.pdf', 'a') as f:
        f.write('revised')
    copy_handbook_files(files, handbooks_dir, str(document_dir))
    assert (document_dir / 'NICMOS-Data-Handbook-2009-v8.pdf').read_text().endswith(
        'revised')

    # Handbooks of instruments no longer in the program, and renamed handbooks, are
    # removed; other files are kept
    (document_dir / 'wfpc1-data-handbook.lblx').write_text('renamed since')
    (_, files) = get_handbook_files(['NICMOS'], handbooks_dir)
    copy_handbook_files(files, handbooks_dir, str(document_dir))
    assert sorted(os.listdir(document_dir)) == sorted(files + ['07885.pro'])


def test_get_handbooks_dir(tmp_path, monkeypatch):
    monkeypatch.delenv('HST_HANDBOOKS', raising=False)
    with pytest.raises(HandbooksError, match='HST_HANDBOOKS'):
        get_handbooks_dir()

    monkeypatch.setenv('HST_HANDBOOKS', str(tmp_path / 'missing'))
    with pytest.raises(HandbooksError, match='not found'):
        get_handbooks_dir()

    # The variable names the bundle, whose document collection holds the handbooks, or
    # else the collection directory itself
    monkeypatch.setenv('HST_HANDBOOKS', str(tmp_path))
    assert get_handbooks_dir() == str(tmp_path)
    (tmp_path / 'document').mkdir()
    assert get_handbooks_dir() == str(tmp_path / 'document')
