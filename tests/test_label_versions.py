##########################################################################################
# tests/test_label_versions.py
#
# Tests for product label versions: relabeling against the previous versions keeps an
# unchanged label, raises the minor version of a changed label whose FITS file is the
# same, and raises the major version of a changed FITS file.
##########################################################################################

import re
import shutil

from astropy.io import fits

from hst_pipeline.hst_helper.general_utils import get_label_version_id
from hst_pipeline.product_labels import (label_bodies_match,
                                         label_hst_fits_directories,
                                         unwrap_description)
from hst_pipeline.product_labels.xml_support import get_modification_history

from .utils import golden_filepath, TEST_COPIES_DIR

VISIT = 'hst_07885/visit_01/mastDownload/HST/n4wl01xdq'


def _visit_copy(tmp_path, name):
    """Copy the golden visit's FITS files to tmp_path/name; return the directory."""
    source = golden_filepath(f'{TEST_COPIES_DIR}/{VISIT}')
    target = tmp_path / name
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('*.lblx'))
    return target


def _label(directory, old=None):
    label_hst_fits_directories([str(directory)],
                               old_directories=[str(old)] if old else [],
                               retrieval_date='2026-01-02')


def _history(label_path):
    return [(version, date) for (date, version, _)
            in get_modification_history(label_path.read_text())]


def test_versions(tmp_path):
    old = _visit_copy(tmp_path, 'old')
    _label(old)
    raw_label = old / 'n4wl01xdq_raw.lblx'
    assert get_label_version_id(str(raw_label)) == (1, 0)
    assert 'retrieved from MAST on 2026-01-02' in raw_label.read_text()

    # Unchanged: every label is kept as it was
    new = _visit_copy(tmp_path, 'new')
    _label(new, old)
    for label in old.glob('*.lblx'):
        assert (new / label.name).read_text() == label.read_text()

    # The previous label differs (as if the labeling software changed): version 1.1,
    # whose history keeps the old entry
    text = raw_label.read_text()
    raw_label.write_text(text.replace('<title>', '<title>Old ', 1))
    _label(new, old)
    new_label = new / 'n4wl01xdq_raw.lblx'
    assert get_label_version_id(str(new_label)) == (1, 1)
    assert [v for (v, _) in _history(new_label)] == [(1, 0), (1, 1)]
    assert 'is a minor update' in new_label.read_text()
    assert (new / 'n4wl01xdq_spt.lblx').read_text() \
        == (old / 'n4wl01xdq_spt.lblx').read_text()

    # The FITS file differs: version 2.0
    with fits.open(new / 'n4wl01xdq_raw.fits', mode='update') as hdulist:
        hdulist[0].header['HISTORY'] = 'reprocessed'
    _label(new, old)
    assert get_label_version_id(str(new_label)) == (2, 0)
    assert 'is a major update' in new_label.read_text()


def test_label_bodies_match():
    label = ('<!-- version 1.0 (2026-01-02) -->\n<version_id>1.0</version_id>\n'
             '<Modification_History>a</Modification_History>\n'
             '<comment>see urn:x:y::1.0</comment>\n')
    other = re.sub(r'1\.0', '1.1', label).replace('>a<', '>b<')
    assert label_bodies_match(label, other)
    assert not label_bodies_match(label, other.replace('see', 'See'))


def test_unwrap_description():
    text = """This is the initial PDS4 version.

          This FITS file was retrieved
          from MAST."""
    assert unwrap_description(text) == ('This is the initial PDS4 version.\n\n'
                                        'This FITS file was retrieved from MAST.')
