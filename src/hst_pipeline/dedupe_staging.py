##########################################################################################
# dedupe_staging.py
#
# dedupe_staging is the main function called in the dedupe_staging script. It removes
# the duplicate copies left in staging by older versions of the pipeline, which copied
# each downloaded file into its collection directory and kept the original under
# <HST_STAGING>/hst_<nnnnn>/visit_<ss>/mastDownload/. A visit's mastDownload tree is
# deleted only after every file in it is verified to be redundant.
##########################################################################################

import filecmp
import os
import shutil
from collections import namedtuple

import pdslogger

from hst_pipeline.hst_helper import MAST_DOWNLOAD_DIRNAME
from hst_pipeline.hst_helper.fs_utils import get_program_dir_path
from hst_pipeline.prepare_browse_products import get_filed_path

# Suffix that older download steps appended to a repeated file instead of dropping it.
DUPLICATE_SUFFIX = '-duplicate'

# Outcome for one visit.
#   visit       two character visit.
#   redundant   number of files with a byte-identical filed copy.
#   unfiled     paths of files that belong in no collection (never filed, so redundant).
#   problems    paths of files whose filed copy is missing or differs.
#   deleted     True if the visit's mastDownload tree was removed.
VisitResult = namedtuple('VisitResult',
                         ['visit', 'redundant', 'unfiled', 'problems', 'deleted'])


def check_visit_inbox(proposal_id, visit):
    """Check every file in one visit's mastDownload tree against its filed copy. A file
    named "<name>-duplicate" is checked against the filed copy of <name>.

    Inputs:
        proposal_id    a proposal id.
        visit          two character visit.

    Returns:    a VisitResult, with deleted False.
    """
    inbox = os.path.join(get_program_dir_path(proposal_id, visit, 'staging'),
                         MAST_DOWNLOAD_DIRNAME)
    redundant = 0
    unfiled = []
    problems = []
    for root, _, files in os.walk(inbox):
        for file in sorted(files):
            path = os.path.join(root, file)
            dest = get_filed_path(proposal_id, visit, file)
            if dest is None and file.endswith(DUPLICATE_SUFFIX):
                # A renamed repeat is redundant only if it matches the filed original.
                dest = (get_filed_path(proposal_id, visit,
                                       file[:-len(DUPLICATE_SUFFIX)]) or '')
            if dest is None:
                unfiled.append(path)
            elif os.path.isfile(dest) and filecmp.cmp(path, dest, shallow=False):
                redundant += 1
            else:
                problems.append(path)
    return VisitResult(visit, redundant, unfiled, problems, False)


def dedupe_staging(proposal_id, delete=False, logger=None):
    """Remove the redundant mastDownload trees of one proposal's staging directory.

    For each visit_<ss>/mastDownload tree, every file must either have a byte-identical
    copy in its collection directory or belong in no collection. Only then, and only if
    delete is True, is the tree removed. Otherwise nothing in the visit is changed, and
    each file that blocked the deletion is logged.

    Inputs:
        proposal_id    a proposal id.
        delete         True to delete verified trees; False to report only.
        logger         pdslogger to use; None for default EasyLogger.

    Returns:    a list of VisitResult, one per visit with a mastDownload tree.
    """
    logger = logger or pdslogger.EasyLogger()
    staging_dir = get_program_dir_path(proposal_id, None, 'staging')
    logger.info(f'Dedupe staging directory {staging_dir} (delete={delete})')

    results = []
    for name in sorted(os.listdir(staging_dir)):
        inbox = os.path.join(staging_dir, name, MAST_DOWNLOAD_DIRNAME)
        if not name.startswith('visit_') or not os.path.isdir(inbox):
            continue

        visit = name[len('visit_'):]
        result = check_visit_inbox(proposal_id, visit)
        for path in result.unfiled:
            logger.warn('Belongs in no collection; never filed', path)
        for path in result.problems:
            logger.error('Filed copy missing or different', path)

        if result.problems:
            logger.warn(f'Visit {visit}: {len(result.problems)} file(s) not verified; '
                        f'{inbox} kept')
        elif delete:
            shutil.rmtree(inbox)
            result = result._replace(deleted=True)
            logger.info(f'Visit {visit}: {result.redundant} redundant file(s); '
                        f'removed {inbox}')
        else:
            logger.info(f'Visit {visit}: {result.redundant} redundant file(s); '
                        f'{inbox} can be removed')
        results.append(result)

    return results
