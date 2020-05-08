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


_STAGE = Callable[[], None]

_COMMAND_DICT: Dict[str, Callable[[Directories, int], _STAGE]] = {
    # Clear everything except for the cached downloads
    "reset_pipeline": ResetPipeline,
    # Download document files.
    "download_docs": DownloadDocs,
    # Download FITS files.
    "check_downloads": CheckDownloads,
    # Copy primary files (documents and FITS data files) into a
    # staging area.
    "copy_primary_files": CopyPrimaryFiles,
    # Note which files have changes and save this information into
    # a file.
    "record_changes": RecordChanges,
    # Insert the actual changes into (a layer over) the archive.
    "insert_changes": InsertChanges,
    # Fill the database with information from new primary files.
    "populate_database": PopulateDatabase,
    # Build a browse collection for each new data collection.
    "build_browse": BuildBrowse,
    # Build labels for the new components.
    "build_labels": BuildLabels,
    # Update the archive with the new version
    "update_archive": UpdateArchive,
    # Build labels for the new components.
    "make_deliverable": MakeDeliverable,
    # Run the validation tool on the deliverable bundle
    "validate_bundle": ValidateBundle,
}


def run() -> None:
    assert len(sys.argv) == 3, sys.argv
    proposal_id = int(sys.argv[1])
    command = sys.argv[2]

    dirs = make_directories()
    _COMMAND_DICT[command](dirs, proposal_id)()


if __name__ == "__main__":
    run()
