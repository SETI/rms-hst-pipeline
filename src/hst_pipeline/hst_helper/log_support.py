##########################################################################################
# hst_helper/log_support.py
#
# Loggers for the pipeline tasks, and the files that flag runs needing review:
# <HST_PIPELINE>/hst_<nnnnn>/WARNINGS.log and <HST_PIPELINE>/NEEDS_REVIEW.log.
##########################################################################################

import datetime
import os
import pdslogger
import sys

from hst_pipeline.hst_helper import HST_DIR
from hst_pipeline.hst_helper.fs_utils import get_formatted_proposal_id

# Every WARNING, ERROR, or FATAL message of a program's tasks is appended to the
# WARNINGS.log in its pipeline directory; for the tasks that belong to no program, to
# the one in <HST_PIPELINE>/logs.
WARNINGS_LOG = 'WARNINGS.log'

# Every task run that logged a warning or worse, and every task the queue manager saw
# fail or killed, adds one line to <HST_PIPELINE>/NEEDS_REVIEW.log. Delete either file
# to acknowledge what it lists; the next problem creates it again.
NEEDS_REVIEW_LOG = 'NEEDS_REVIEW.log'

LOG_LIMITS = {'info': -1, 'debug': -1, 'normal': -1}

def get_log_dir(proposal_id=None, visit=None):
    """Return the directory of a task's timestamped logs.

    Inputs:
        proposal_id    a proposal id; None for a task that belongs to no program.
        visit          two character visit; None for the program's own logs directory.

    Returns:    <HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/logs,
                <HST_PIPELINE>/hst_<nnnnn>/logs, or <HST_PIPELINE>/logs.
    """
    root = HST_DIR['pipeline']
    if not proposal_id:
        return f'{root}/logs'
    program_dir = f'{root}/hst_{get_formatted_proposal_id(proposal_id)}'
    if visit:
        return f'{program_dir}/visit_{str(visit).zfill(2)}/logs'
    return f'{program_dir}/logs'

def get_warnings_dir(proposal_id=None):
    """Return the directory holding the WARNINGS.log that a task's warnings go to.

    Input:
        proposal_id    a proposal id; None for a task that belongs to no program.

    Returns:    <HST_PIPELINE>/hst_<nnnnn>, or <HST_PIPELINE>/logs.
    """
    if not proposal_id:
        return get_log_dir()
    return f'{HST_DIR["pipeline"]}/hst_{get_formatted_proposal_id(proposal_id)}'

def get_needs_review_path():
    """Return the path of <HST_PIPELINE>/NEEDS_REVIEW.log."""
    return os.path.join(HST_DIR['pipeline'], NEEDS_REVIEW_LOG)

def setup_task_logger(task, args, argv=None, proposal_id=None, visit=None,
                      visit_logs=False):
    """Create and open the logger of a pipeline task.

    It logs to the terminal unless args.quiet, to a timestamped log file, and, at the
    WARNING level and above, to the WARNINGS.log of the program (see get_warnings_dir).
    Close it with close_task_logger().

    Inputs:
        task           the task's command name without the "hst-" prefix, e.g.,
                       'finalize-hst-bundle'; it starts the log's file name.
        args           the parsed arguments; uses args.quiet and args.log (a log file
                       name, to which the time is appended, or a directory; '' for the
                       default directory).
        argv           the command-line arguments; None for sys.argv[1:].
        proposal_id    the task's proposal id; None if it belongs to no program.
        visit          the task's visit, if any.
        visit_logs     True to put the log in the visit's logs directory rather than
                       the program's.

    Returns:    the open PdsLogger.
    """
    name = f'pds.hst.{task}'
    if proposal_id:
        name += f'-{proposal_id}'
    if visit:
        name += f'-visit_{str(visit).zfill(2)}'
    logger = pdslogger.PdsLogger(name)
    if not args.quiet:
        logger.add_handler(pdslogger.stdout_handler)

    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if args.log:
        if os.path.isdir(args.log):
            logpath = os.path.join(args.log, f'{task}-{now}.log')
        else:
            parts = os.path.splitext(args.log)
            logpath = parts[0] + '-' + now + parts[1]
    else:
        log_dir = get_log_dir(proposal_id, visit if visit_logs else None)
        os.makedirs(log_dir, exist_ok=True)
        logpath = f'{log_dir}/{task}-{now}.log'
    logger.add_handler(pdslogger.file_handler(logpath))

    warnings_dir = get_warnings_dir(proposal_id)
    os.makedirs(warnings_dir, exist_ok=True)
    logger.add_handler(pdslogger.warning_handler(warnings_dir, name=WARNINGS_LOG))

    cmd_args = sys.argv[1:] if argv is None else argv
    title = ' '.join([task] + list(cmd_args))
    logger.hst_task_info = (title, logpath)
    logger.open(title, limits=LOG_LIMITS)
    return logger

def close_task_logger(logger):
    """Close a logger made by setup_task_logger, and add a line to NEEDS_REVIEW.log if
    the task logged a warning or worse.

    Input:
        logger    the logger.

    Returns:    a tuple (fatal, errors, warnings, total) of message counts.
    """
    counts = logger.close()
    (fatal, errors, warnings, _) = counts
    if fatal or errors or warnings:
        (title, logpath) = logger.hst_task_info
        record_needs_review(title, format_counts(fatal, errors, warnings), logpath)
    return counts

def format_counts(fatal, errors, warnings):
    """Return the nonzero message counts as text, e.g., "2 ERROR, 5 WARNING"."""
    parts = [f'{count} {name}' for (count, name) in [(fatal, 'FATAL'),
                                                       (errors, 'ERROR'),
                                                       (warnings, 'WARNING')]
             if count]
    return ', '.join(parts)

def record_needs_review(title, problem, path=''):
    """Append one line to <HST_PIPELINE>/NEEDS_REVIEW.log:
    "<time> | <title> | <problem> | <path>".

    Inputs:
        title      the task, e.g., "finalize-hst-bundle --proposal-id 10992".
        problem    what needs review, e.g., "1 ERROR, 2 WARNING".
        path       the file to look at, usually the task's log; '' for none.
    """
    now = datetime.datetime.now().isoformat(timespec='seconds')
    line = ' | '.join([now, title, problem, path]).rstrip(' |') + '\n'
    path = get_needs_review_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(line)
