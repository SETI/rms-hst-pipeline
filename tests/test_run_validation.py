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
    _write(bundle / 'bundle.lblx')
    _write(bundle / 'data_nicmos_raw' / 'collection_data_nicmos_raw.lblx')
    _write(bundle / 'data_nicmos_raw' / 'visit_01' / 'n4wl01xdq_raw.lblx')

    run_validation.create_manifest_files(7885, None)

    program_dir = tmp_path / 'hst_07885-deliverable'
    checksums = (program_dir / run_validation.CM_FNAME).read_text().splitlines()
    assert [line.split('  ')[1] for line in checksums] == [
        'bundle.lblx',
        'data_nicmos_raw/collection_data_nicmos_raw.lblx',
        'data_nicmos_raw/visit_01/n4wl01xdq_raw.lblx',
    ]

    transfers = [line.split() for line in
                 (program_dir / run_validation.TM_FNAME).read_text().splitlines()]
    assert transfers == [
        ['urn:nasa:pds:hst_07885::1.0', 'bundle.lblx'],
        ['urn:nasa:pds:hst_07885:data_nicmos_raw::1.0',
         'data_nicmos_raw/collection_data_nicmos_raw.lblx'],
        ['urn:nasa:pds:hst_07885:data_nicmos_raw:n4wl01xdq_raw::1.0',
         'data_nicmos_raw/visit_01/n4wl01xdq_raw.lblx'],
    ]


def test_context_json_is_written_where_validation_reads_it(tmp_path, monkeypatch):
    # create_target_label writes the file and run_validation passes the same path to
    # validate-pdart, whatever the current directory is.
    from hst_pipeline import create_target_label

    monkeypatch.setitem(fs_utils.__dict__, 'HST_DIR', {'bundles': str(tmp_path)})
    monkeypatch.chdir(tmp_path)
    create_target_label.create_tmp_context_json(14609, {'inst_id_li': ['COS']})

    expected = tmp_path / 'hst_14609-deliverable' / 'tmp-context-products-14609.json'
    assert expected.is_file()
    assert fs_utils.get_context_products_path(14609) == str(expected)
    assert not list(tmp_path.glob('tmp-context-products-*'))   # not in the cwd

    calls = []
    monkeypatch.setattr(run_validation, 'create_manifest_files', lambda *args: None)
    monkeypatch.setattr(run_validation, 'run', lambda args, **kwargs: calls.append(args))
    run_validation.run_validation(14609)

    assert calls[0][1:] == [str(tmp_path / 'hst_14609-deliverable' / 'hst_14609'),
                            str(tmp_path / 'hst_14609-deliverable'),
                            str(tmp_path / 'hst_14609-deliverable'),
                            str(expected)]
    assert not expected.exists()     # removed after validation
