##########################################################################################
# run_validation/__init__.py
##########################################################################################
import os
import pdslogger
from subprocess import run

from hst_helper.fs_utils import (get_deliverable_path,
                                 get_program_dir_path,
                                 get_formatted_proposal_id,
                                 file_md5)

CM_FNAME = 'checksum.manifest.txt'
TM_FNAME = 'transfer.manifest.txt'

VID = '1.0'

def run_validation(proposal_id, logger=None):
    """Run validator and generate report.
    1. Create checksum_manifest.txt & transfer.manifest.txt
    2. run validate

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Run validation for proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    create_manifest_files(proposal_id, logger)

    bundle_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    print('0-0-0-0-0-0-00-0-0-0-')
    print(bundle_dir)
    run(["./validate-pdart", bundle_dir, bundle_dir, bundle_dir])


def create_manifest_files(proposal_id, logger):
    """With a given proposal id, create checksum manifest and transfer manifest files.

    Inputs:
        proposal_id:    a proposal id.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Create manifest files for proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    cm_path = get_program_dir_path(proposal_id, None, 'bundles') + f'/{CM_FNAME}'
    tm_path = get_program_dir_path(proposal_id, None, 'bundles') + f'/{TM_FNAME}'
    deliverable_path = get_deliverable_path(proposal_id)
    cm_files_li = set()
    tm_files_li = set()
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    lidvid_prefix = f'urn:nasa:pds:hst_{formatted_proposal_id}'
    for root, dirs, files in os.walk(deliverable_path, topdown=True):
        for name in files:
            file_path = os.path.join(root, name)
            _, _, file_logical_path = file_path.partition('deliverable/')
            cm_files_li.add((file_path, file_logical_path))
            if 'bundle' in file_logical_path:
                lidvid = lidvid_prefix + f'::{VID}'
            elif 'collection' in file_logical_path:
                col_name, _, _ = file_logical_path.partition('/')
                lidvid = lidvid_prefix + f'{col_name}::{VID}'
            elif 'individual' not in file_logical_path:
                try:
                    col_name, _, fname = file_logical_path.rpartition('.')[0].split('/')
                except ValueError:
                    continue # ignore files like .DS_Store
                lidvid = lidvid_prefix + f'{col_name}:{fname}::{VID}'
            tm_files_li.add((lidvid, file_logical_path))

    cm_files_li = sorted(cm_files_li)
    with open(cm_path, 'w') as f:
        for fp, logical_fp in cm_files_li:
                checksum = file_md5(fp)
                f.write('%s  %s\n' % (checksum, logical_fp))

    tm_files_li = sorted(tm_files_li)
    max_width = max(len(lidvid) for (lidvid, _) in tm_files_li)
    with open(tm_path, 'w') as f:
        for lidvid, logical_fp in tm_files_li:
            f.write('%-*s %s\n' % (max_width, lidvid, logical_fp))
