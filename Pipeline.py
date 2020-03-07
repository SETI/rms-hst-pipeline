import sys

import os.path
import traceback

from pdart.pipeline.BuildBrowse import build_browse
from pdart.pipeline.BuildLabels import build_labels
from pdart.pipeline.CheckDownloads import check_downloads
from pdart.pipeline.CopyPrimaryFiles import copy_primary_files
from pdart.pipeline.Directories import Directories
from pdart.pipeline.DownloadDocs import download_docs
from pdart.pipeline.InsertChanges import insert_changes
from pdart.pipeline.PopulateDatabase import populate_database
from pdart.pipeline.RecordChanges import record_changes


def dispatch(dirs, proposal_id, command):
    # type: (Directories, int, str) -> None
    bundle_segment = "hst_%05d" % proposal_id

    # Here's a list of all the commands available.
    command_dict = {
        # Download document files.
        "download_docs": (
            lambda: download_docs(dirs.documents_dir(proposal_id), proposal_id)
        ),
        # Download FITS files.
        "check_downloads": (
            lambda: check_downloads(
                dirs.working_dir(proposal_id),
                dirs.mast_downloads_dir(proposal_id),
                proposal_id,
            )
        ),
        # Copy primary files (documents and FITS data files) into a
        # staging area.
        "copy_primary_files": (
            lambda: copy_primary_files(
                bundle_segment,
                dirs.documents_dir(proposal_id),
                dirs.mast_downloads_dir(proposal_id),
                dirs.primary_files_dir(proposal_id),
            )
        ),
        # Note which files have changes and save this information into
        # a file.
        "record_changes": (
            lambda: record_changes(
                bundle_segment,
                dirs.working_dir(proposal_id),
                dirs.primary_files_dir(proposal_id),
                dirs.archive_dir(proposal_id),
            )
        ),
        # Insert the actual changes into (a layer over) the archive.
        "insert_changes": (
            lambda: insert_changes(
                bundle_segment,
                dirs.working_dir(proposal_id),
                dirs.primary_files_dir(proposal_id),
                dirs.archive_dir(proposal_id),
                dirs.archive_primary_deltas_dir(proposal_id),
            )
        ),
        # Fill the database with information from new primary files.
        "populate_database": (
            lambda: populate_database(
                bundle_segment,
                dirs.working_dir(proposal_id),
                dirs.archive_dir(proposal_id),
                dirs.archive_primary_deltas_dir(proposal_id),
            )
        ),
        # Build a browse collection for each new data collection.
        "build_browse": (
            lambda: build_browse(
                bundle_segment,
                dirs.working_dir(proposal_id),
                dirs.archive_dir(proposal_id),
                dirs.archive_primary_deltas_dir(proposal_id),
                dirs.archive_browse_deltas_dir(proposal_id),
            )
        ),
        # Build labels for the new components.
        "build_labels": (
            lambda: build_labels(
                bundle_segment,
                dirs.working_dir(proposal_id),
                dirs.archive_dir(proposal_id),
                dirs.archive_primary_deltas_dir(proposal_id),
                dirs.archive_browse_deltas_dir(proposal_id),
                dirs.archive_label_deltas_dir(proposal_id),
            )
        ),
    }
    command_dict[command]()


_FAILURE_MARKER = "LAST$FAILURE.txt"


def run():
    assert len(sys.argv) == 3, sys.argv
    proposal_id = int(sys.argv[1])
    command = sys.argv[2]

    dirs = Directories("tmp-working-dir")
    failure_marker_filepath = os.path.join(
        dirs.working_dir(proposal_id), _FAILURE_MARKER
    )

    if os.path.isfile(failure_marker_filepath):
        print "**** Skipping %d %s due to previous failures." % (proposal_id, command)
        return

    try:
        dispatch(dirs, proposal_id, command)
    except Exception as e:
        with open(failure_marker_filepath, "w") as f:
            header = "EXCEPTION raised by hst_%05d, stage %s:" % (proposal_id, command)
            print "****", header, str(e)
            print >> f, header
            print >> f, traceback.format_exc()


if __name__ == "__main__":
    run()
