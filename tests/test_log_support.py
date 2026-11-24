##########################################################################################
# tests/test_log_support.py
#
# Tests for the task loggers and the files that flag runs needing review: each program's
# WARNINGS.log, <HST_PIPELINE>/NEEDS_REVIEW.log, the queue manager's failed and killed
# tasks, and the messages of validate's report.
##########################################################################################

import argparse
import itertools

import pdslogger
import pytest

from hst_pipeline import queue_manager
from hst_pipeline import run_validation
from hst_pipeline.hst_helper import log_support

# Each PdsLogger name is used once, since Python's loggers outlive the tests
_IDS = itertools.count(90001)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Point HST_PIPELINE at tmp_path; return it."""
    monkeypatch.setitem(log_support.HST_DIR, 'pipeline', str(tmp_path))
    return tmp_path


def _args(log=''):
    return argparse.Namespace(quiet=True, log=log)


def _review_lines(pipeline):
    path = pipeline / log_support.NEEDS_REVIEW_LOG
    return path.read_text().splitlines() if path.exists() else []


def test_warnings_go_to_the_program_and_the_run_to_needs_review(pipeline):
    pid = str(next(_IDS))
    logger = log_support.setup_task_logger('finalize-hst-bundle', _args(),
                                           ['--proposal-id', pid], proposal_id=pid)
    logger.info('routine')
    logger.warn('look at this', '/a/b.fits')
    logger.error('broken')
    assert log_support.close_task_logger(logger)[:3] == (0, 1, 1)

    (log,) = (pipeline / f'hst_{pid}' / 'logs').iterdir()
    assert log.name.startswith('finalize-hst-bundle-')
    warnings = (pipeline / f'hst_{pid}' / 'WARNINGS.log').read_text().splitlines()
    assert len(warnings) == 2
    assert '| WARNING | look at this: /a/b.fits' in warnings[0]
    assert '| ERROR | broken' in warnings[1]
    assert all(f'finalize-hst-bundle-{pid}' in line for line in warnings)

    (line,) = _review_lines(pipeline)
    assert line.split(' | ')[1:] == [f'finalize-hst-bundle --proposal-id {pid}',
                                     '1 ERROR, 1 WARNING', str(log)]


def test_clean_run_leaves_no_trace(pipeline):
    pid = str(next(_IDS))
    logger = log_support.setup_task_logger('label-hst-products', _args(), [],
                                           proposal_id=pid, visit='3', visit_logs=True)
    logger.info('all fine')
    log_support.close_task_logger(logger)

    assert len(list((pipeline / f'hst_{pid}' / 'visit_03' / 'logs').iterdir())) == 1
    assert not (pipeline / f'hst_{pid}' / 'WARNINGS.log').exists() or \
        (pipeline / f'hst_{pid}' / 'WARNINGS.log').read_text() == ''
    assert _review_lines(pipeline) == []


def test_task_without_program_and_log_option(pipeline, tmp_path):
    other = tmp_path / 'elsewhere'
    other.mkdir()
    logger = log_support.setup_task_logger(f'query-hst-moving-targets-{next(_IDS)}',
                                           _args(log=str(other)), [])
    logger.warn('odd')
    log_support.close_task_logger(logger)

    assert len(list(other.iterdir())) == 1      # the --log directory holds the log
    assert 'odd' in (pipeline / 'logs' / 'WARNINGS.log').read_text()
    (line,) = _review_lines(pipeline)
    assert str(other) in line


def test_log_dirs(pipeline):
    assert log_support.get_log_dir() == f'{pipeline}/logs'
    assert log_support.get_log_dir(7885) == f'{pipeline}/hst_07885/logs'
    assert log_support.get_log_dir('7885', '1') == f'{pipeline}/hst_07885/visit_01/logs'
    assert log_support.get_warnings_dir('07885') == f'{pipeline}/hst_07885'
    assert log_support.get_warnings_dir() == f'{pipeline}/logs'


def test_format_counts_and_record(pipeline):
    assert log_support.format_counts(1, 0, 3) == '1 FATAL, 3 WARNING'
    assert log_support.format_counts(0, 2, 0) == '2 ERROR'
    log_support.record_needs_review('hst-retrieve-visit --proposal-id 5167', 'exit 1')
    (line,) = _review_lines(pipeline)
    assert line.endswith(' | hst-retrieve-visit --proposal-id 5167 | exit 1')


##########################################################################################
# queue manager
##########################################################################################

class _Proc:
    def __init__(self, returncode):
        self.returncode = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True


def _logger():
    logger = pdslogger.PdsLogger(f'test.queue-{next(_IDS)}')
    logger.open('test')
    return logger


@pytest.mark.parametrize('returncode', [0, 1])
def test_queue_records_failed_task(pipeline, monkeypatch, returncode):
    args = ['/venv/bin/hst-finalize-bundle', '--proposal-id', '05167']
    monkeypatch.setattr(queue_manager, 'SUBPROCESS_LIST',
                        [(_Proc(returncode), 0, 1e20, '05167', '', 'finalize_bundle',
                          args)])
    logger = _logger()
    queue_manager.wait_for_subprocess(logger, all=True)

    assert logger.close()[1] == returncode
    lines = _review_lines(pipeline)
    if returncode:
        (line,) = lines
        assert line.endswith(' | hst-finalize-bundle --proposal-id 05167 | exit status 1')
    else:
        assert lines == []


def test_queue_records_killed_task(pipeline, monkeypatch):
    proc = _Proc(None)
    args = ['/venv/bin/hst-retrieve-visit', '--proposal-id', '05167', '--visit', '01']
    monkeypatch.setattr(queue_manager, 'SUBPROCESS_LIST',
                        [(proc, 0, 0, '05167', '01', 'retrieve_visit', args)])
    monkeypatch.setattr(queue_manager, 'remove_a_task', lambda *args: None)
    logger = _logger()
    queue_manager.wait_for_subprocess(logger, all=True)

    assert proc.killed
    assert logger.close()[1] == 1
    (line,) = _review_lines(pipeline)
    assert 'hst-retrieve-visit --proposal-id 05167 --visit 01 | killed after' in line


##########################################################################################
# validation report
##########################################################################################

REPORT = """\
Product Level Validation Results

  PASS: gov.nasa.pds.validate.ValidateLauncher (N/A)
      WARNING  [warning.product_not_registered]   Non-registered context products.

  FAIL: file:/b/hst_05167/document/hst-data-handbook-volume-2.lblx (urn:x::1.0)
      ERROR  [error.label.context_ref_not_found]   line 97: 'urn:nasa:pds:context:x'
      WARNING  [warning.label.context_ref_mismatch]   line 70: name mismatch

  PASS: file:/b/hst_05167/bundle.lblx (urn:y::1.0)

  Message Types:
    1            warning.product_not_registered
"""


def test_validation_report_messages(tmp_path):
    report = tmp_path / 'validation-report.txt'
    report.write_text(REPORT)
    logger = _logger()
    path = tmp_path / 'task.log'
    logger.add_handler(pdslogger.file_handler(str(path)))

    assert run_validation.log_validation_report(str(report), logger) == (1, 1)
    assert logger.close()[1:3] == (1, 1)

    text = path.read_text()
    label = '/b/hst_05167/document/hst-data-handbook-volume-2.lblx'
    assert ("| ERROR | validate [error.label.context_ref_not_found] line 97: "
            f"'urn:nasa:pds:context:x': {label}") in text
    assert '| WARNING | validate [warning.label.context_ref_mismatch]' in text
    # Expected during development, so not a warning
    assert '| INFO | validate [warning.product_not_registered]' in text


def test_validation_report_missing(tmp_path):
    logger = _logger()
    assert run_validation.log_validation_report(str(tmp_path / 'none.txt'), logger) \
        == (1, 0)
    assert logger.close()[1] == 1
