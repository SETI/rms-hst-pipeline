##########################################################################################
# run_validation.py
##########################################################################################

import os
import pdslogger
import re
import shutil
from subprocess import run

from hst_pipeline.hst_helper.fs_utils import (get_context_products_path,
                                              get_deliverable_path,
                                              get_program_dir_path,
                                              get_formatted_proposal_id,
                                              file_md5)

CM_FNAME = 'checksum.manifest.txt'
TM_FNAME = 'transfer.manifest.txt'

HST_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATE_PDART = os.path.join(HST_DIR, 'validate-pdart')

VID = '1.0'

REPORT_FNAME = 'validation-report.txt'

# Validate's messages that are expected of every bundle during development, and are
# logged at INFO rather than WARNING: the pipeline's context products aren't registered.
EXPECTED_VALIDATE_MESSAGES = {'warning.product_not_registered'}

# A product line ("  FAIL: file:/path/x.lblx (urn:...)") and a message line
# ("      ERROR  [error.label.context_ref_not_found]   line 97: ...") of the report
_PRODUCT_LINE = re.compile(r'\s*(?:PASS|FAIL|SKIP): (\S+)')
_MESSAGE_LINE = re.compile(r'\s*(FATAL_ERROR|ERROR|WARNING|INFO)\s+\[([^\]]+)\]\s*(.*)')

def run_validation(proposal_id, logger=None):
    """Run validator and generate report.
    1. Create checksum_manifest.txt & transfer.manifest.txt
    2. run validate

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Run validation for proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError: #pragma: no cover
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    create_manifest_files(proposal_id, logger)

    # Validate the bundle itself; the manifests and the report are in its parent, the
    # hst_<nnnnn>-deliverable program directory.
    bundle_dir = get_deliverable_path(proposal_id)
    program_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    logger.info(f'Run validation-pdart for proposal id: {proposal_id} on {bundle_dir}')
    try:
        run([VALIDATE_PDART, bundle_dir, program_dir, program_dir,
             get_context_products_path(proposal_id)],
            cwd=HST_DIR, check=True)
    except:
        logger.error(
            f'Check validation-pdart failure report for proposal id: {proposal_id} '
            f'at {program_dir}/{REPORT_FNAME}')

    log_validation_report(f'{program_dir}/{REPORT_FNAME}', logger)

    # remove tmp context json
    try:
        os.remove(get_context_products_path(proposal_id))
    except FileNotFoundError: #pragma: no cover
        pass

def log_validation_report(report_path, logger):
    """Log each message of a validation report at its own level, with the product it
    concerns: FATAL_ERROR and ERROR as errors, WARNING as warnings, except that the
    messages in EXPECTED_VALIDATE_MESSAGES are logged as info.

    Inputs:
        report_path    path to the validation report.
        logger         pdslogger to use.

    Returns:    a tuple (errors, warnings), the numbers of messages logged as errors and
                as warnings.
    """
    try:
        with open(report_path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        logger.error('Validation report not found', report_path)
        return (1, 0)

    errors = 0
    warnings = 0
    product = ''
    for line in lines:
        if match := _PRODUCT_LINE.match(line):
            product = match.group(1).removeprefix('file:')
            continue
        if not (match := _MESSAGE_LINE.match(line)):
            continue
        (level, msg_type, text) = match.groups()
        message = f'validate [{msg_type}] {text}'.rstrip()
        if msg_type in EXPECTED_VALIDATE_MESSAGES or level == 'INFO':
            logger.info(message, product)
        elif level == 'WARNING':
            logger.warn(message, product)
            warnings += 1
        else:
            logger.error(message, product)
            errors += 1

    logger.info(f'Validation report has {errors} error(s) and {warnings} warning(s) '
                'needing review', report_path)
    return (errors, warnings)

def create_manifest_files(proposal_id, logger):
    """With a given proposal id, create checksum manifest and transfer manifest files.

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Create manifest files for proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError: #pragma: no cover
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    cm_path = f'{get_program_dir_path(proposal_id, None, "bundles")}/{CM_FNAME}'
    tm_path = f'{get_program_dir_path(proposal_id, None, "bundles")}/{TM_FNAME}'
    deliverable_path = get_deliverable_path(proposal_id)
    cm_files_li = set()
    tm_files_li = set()
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    lidvid_prefix = f'urn:nasa:pds:hst_{formatted_proposal_id}'
    for root, dirs, files in os.walk(deliverable_path, topdown=True):
        for name in files:
            file_path = os.path.join(root, name)
            file_logical_path = os.path.relpath(file_path, deliverable_path)
            cm_files_li.add((file_path, file_logical_path))
            if 'bundle' in file_logical_path:
                lidvid = f'{lidvid_prefix}::{VID}'
            elif 'collection' in file_logical_path:
                col_name, _, _ = file_logical_path.partition('/')
                lidvid = f'{lidvid_prefix}:{col_name}::{VID}'
            elif 'individual' not in file_logical_path:
                try:
                    col_name, _, fname = file_logical_path.rpartition('.')[0].split('/')
                except ValueError: #pragma: no cover
                    continue # ignore files like .DS_Store
                lidvid = f'{lidvid_prefix}:{col_name}:{fname}::{VID}'
            tm_files_li.add((lidvid, file_logical_path))

    cm_files_li = sorted(cm_files_li)
    with open(cm_path, 'w') as f:
        for file_path, logical_file_path in cm_files_li:
                checksum = file_md5(file_path)
                f.write('%s  %s\n' % (checksum, logical_file_path))

    tm_files_li = sorted(tm_files_li)
    max_width = max(len(lidvid) for (lidvid, _) in tm_files_li)
    with open(tm_path, 'w') as f:
        for lidvid, logical_file_path in tm_files_li:
            f.write('%-*s %s\n' % (max_width, lidvid, logical_file_path))
