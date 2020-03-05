import os.path
import sys
import traceback

from pdart.pipeline.BuildBrowse import build_browse
from pdart.pipeline.BuildLabels import build_labels
from pdart.pipeline.CheckDownloads import check_downloads
from pdart.pipeline.CopyDownloads import copy_downloads
from pdart.pipeline.DownloadDocs import download_docs
from pdart.pipeline.CopyPrimaryFiles import copy_primary_files
from pdart.pipeline.Directories import Directories
from pdart.pipeline.InsertChanges import insert_changes
from pdart.pipeline.MakeNewVersions import make_new_versions
from pdart.pipeline.MakeBrowseFiles import make_browse_files
from pdart.pipeline.PopulateDatabase import populate_database
from pdart.pipeline.RecordChanges import record_changes

def dispatch(dirs, proposal_id, command):
    # type: (Directories, int, str) -> None
    bundle_segment = 'hst_%05d' % proposal_id
    if command == 'check_downloads':
        check_downloads(dirs.working_dir(proposal_id),
                        dirs.mast_downloads_dir(proposal_id),
                        proposal_id)
    elif command == 'download_docs':
        download_docs(dirs.documents_dir(proposal_id), proposal_id)
    elif command == 'copy_downloads':
        copy_downloads(bundle_segment,
                       dirs.mast_downloads_dir(proposal_id),
                       dirs.next_version_deltas_dir(proposal_id),
                       dirs.archive_dir(proposal_id))
    elif command == 'copy_primary_files':
        copy_primary_files(bundle_segment,
                           dirs.documents_dir(proposal_id),
                           dirs.mast_downloads_dir(proposal_id),
                           dirs.primary_files_dir(proposal_id))
    elif command == 'make_new_versions':
        make_new_versions(
            bundle_segment, 
            dirs.archive_dir(proposal_id),
            dirs.next_version_deltas_dir(proposal_id),
            dirs.archive_primary_deltas_dir(proposal_id))
    elif command == 'make_browse':
        make_browse_files(
            dirs.archive_dir(proposal_id),
            dirs.archive_primary_deltas_dir(proposal_id),
            dirs.archive_browse_deltas_dir(proposal_id))
    elif command == 'record_changes':
        record_changes(
            bundle_segment,
            dirs.working_dir(proposal_id),
            dirs.primary_files_dir(proposal_id),
            dirs.archive_dir(proposal_id))
    elif command == 'insert_changes':
        insert_changes(
            bundle_segment,
            dirs.working_dir(proposal_id),
            dirs.primary_files_dir(proposal_id),
            dirs.archive_dir(proposal_id),
            dirs.archive_primary_deltas_dir(proposal_id))
    elif command == 'build_browse':
        build_browse(bundle_segment,
                     dirs.working_dir(proposal_id),
                     dirs.archive_dir(proposal_id),
                     dirs.archive_primary_deltas_dir(proposal_id),
                     dirs.archive_browse_deltas_dir(proposal_id))
    elif command == 'build_labels':
        build_labels(bundle_segment,
                     dirs.working_dir(proposal_id),
                     dirs.archive_dir(proposal_id),
                     dirs.archive_primary_deltas_dir(proposal_id),
                     dirs.archive_browse_deltas_dir(proposal_id),
                     dirs.archive_label_deltas_dir(proposal_id))
    elif command == 'populate_database':
        populate_database(bundle_segment,
                          dirs.working_dir(proposal_id),
                          dirs.archive_dir(proposal_id),
                          dirs.archive_primary_deltas_dir(proposal_id))
    else:
        sys.exit(1)

_FAILURE_MARKER = 'LAST$FAILURE.txt'

def run():
    assert(len(sys.argv) == 3), sys.argv
    proposal_id = int(sys.argv[1])
    command = sys.argv[2]

    dirs = Directories('tmp-working-dir')
    failure_marker_filepath = os.path.join(dirs.working_dir(proposal_id),
                                           _FAILURE_MARKER)

    if os.path.isfile(failure_marker_filepath):
        print '**** Skipping %d %s due to previous failures.' % \
            (proposal_id, command)
        return

    try:
        dispatch(dirs, proposal_id, command)
    except Exception as e:
        with open(failure_marker_filepath, 'w') as f:
            f.write(traceback.format_exc())
            print '**** EXCEPTION raised by %d %s: %s' % \
                (proposal_id, command, str(e))
        
if __name__ == '__main__':
    run()
