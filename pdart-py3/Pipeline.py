import os.path
import sys
import traceback
from typing import Callable, Dict

from pdart.pipeline.BuildBrowse import BuildBrowse
from pdart.pipeline.BuildLabels import BuildLabels
from pdart.pipeline.CheckDownloads import CheckDownloads
from pdart.pipeline.CopyPrimaryFiles import CopyPrimaryFiles
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.DownloadDocs import DownloadDocs
from pdart.pipeline.InsertChanges import InsertChanges
from pdart.pipeline.MakeDeliverable import MakeDeliverable
from pdart.pipeline.PopulateDatabase import PopulateDatabase
from pdart.pipeline.RecordChanges import RecordChanges
from pdart.pipeline.ResetPipeline import ResetPipeline
from pdart.pipeline.UpdateArchive import UpdateArchive
from pdart.pipeline.ValidateBundle import ValidateBundle


def dispatch(dirs: Directories, proposal_id: int, command: str) -> None:
    bundle_segment = f"hst_{proposal_id:05}"

    # Here's a list of all the commands available.
    command_dict: Dict[str, Callable[[], None]] = {
        # Clear everything except for the cached downloads
        "reset_pipeline": ResetPipeline(bundle_segment, dirs, proposal_id),
        # Download document files.
        "download_docs": DownloadDocs(bundle_segment, dirs, proposal_id),
        # Download FITS files.
        "check_downloads": CheckDownloads(bundle_segment, dirs, proposal_id),
        # Copy primary files (documents and FITS data files) into a
        # staging area.
        "copy_primary_files": CopyPrimaryFiles(bundle_segment, dirs, proposal_id),
        # Note which files have changes and save this information into
        # a file.
        "record_changes": RecordChanges(bundle_segment, dirs, proposal_id),
        # Insert the actual changes into (a layer over) the archive.
        "insert_changes": InsertChanges(bundle_segment, dirs, proposal_id),
        # Fill the database with information from new primary files.
        "populate_database": PopulateDatabase(bundle_segment, dirs, proposal_id),
        # Build a browse collection for each new data collection.
        "build_browse": BuildBrowse(bundle_segment, dirs, proposal_id),
        # Build labels for the new components.
        "build_labels": BuildLabels(bundle_segment, dirs, proposal_id),
        # Update the archive with the new version
        "update_archive": UpdateArchive(bundle_segment, dirs, proposal_id),
        # Build labels for the new components.
        "make_deliverable": MakeDeliverable(bundle_segment, dirs, proposal_id),
        # Run the validation tool on the deliverable bundle
        "validate_bundle": ValidateBundle(bundle_segment, dirs, proposal_id),
    }
    command_dict[command]()


_FAILURE_MARKER: str = "LAST$FAILURE.txt"


def run() -> None:
    assert len(sys.argv) == 3, sys.argv
    proposal_id = int(sys.argv[1])
    command = sys.argv[2]

    dirs = make_directories()
    failure_marker_filepath = os.path.join(
        dirs.working_dir(proposal_id), _FAILURE_MARKER
    )

    if os.path.isfile(failure_marker_filepath):
        print(f"**** Skipping hst_{proposal_id:05} {command} due to previous failures.")
        return

    try:
        dispatch(dirs, proposal_id, command)
    except Exception as e:
        with open(failure_marker_filepath, "w") as f:
            header = f"EXCEPTION raised by hst_{proposal_id:05}, stage {command}:"
            print("****", header, str(e))
            print(header, file=f)
            print(traceback.format_exc(), file=f)


if __name__ == "__main__":
    run()
