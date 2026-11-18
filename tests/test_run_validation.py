##########################################################################################
# tests/test_run_validation.py
#
# Tests for the manifest files that run_validation writes beside the bundle, in
# <HST_BUNDLES>/hst_<nnnnn>-deliverable/.
##########################################################################################

import os

from hst_pipeline import run_validation
from hst_pipeline.hst_helper import fs_utils


def _write(path, content=b'x'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(content)


def test_manifests_list_paths_relative_to_bundle(tmp_path, monkeypatch):
    monkeypatch.setitem(fs_utils.__dict__, 'HST_DIR', {'bundles': str(tmp_path)})
    bundle = tmp_path / 'hst_07885-deliverable' / 'hst_07885'
    _write(bundle / 'bundle.xml')
    _write(bundle / 'data_nicmos_raw' / 'collection_data_nicmos_raw.xml')
    _write(bundle / 'data_nicmos_raw' / 'visit_01' / 'n4wl01xdq_raw.xml')

    run_validation.create_manifest_files(7885, None)

    program_dir = tmp_path / 'hst_07885-deliverable'
    checksums = (program_dir / run_validation.CM_FNAME).read_text().splitlines()
    assert [line.split('  ')[1] for line in checksums] == [
        'bundle.xml',
        'data_nicmos_raw/collection_data_nicmos_raw.xml',
        'data_nicmos_raw/visit_01/n4wl01xdq_raw.xml',
    ]

    transfers = [line.split() for line in
                 (program_dir / run_validation.TM_FNAME).read_text().splitlines()]
    assert transfers == [
        ['urn:nasa:pds:hst_07885::1.0', 'bundle.xml'],
        ['urn:nasa:pds:hst_07885:data_nicmos_raw::1.0',
         'data_nicmos_raw/collection_data_nicmos_raw.xml'],
        ['urn:nasa:pds:hst_07885:data_nicmos_raw:n4wl01xdq_raw::1.0',
         'data_nicmos_raw/visit_01/n4wl01xdq_raw.xml'],
    ]
