import os.path
import shutil

import fs.path

from pdart.pds4.HstFilename import HstFilename
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import *
from pdart.pipeline.SuffixInfo import get_collection_type  # type: ignore
from pdart.Logging import PDS_LOGGER

_OLD_IMPL = False


def to_segment_dir(name: str) -> str:
    return name + "$"


class CopyPrimaryFiles(MarkedStage):
    """
    Documentation files and data files are "primary files" (as opposed
    to browse files or label files that are derived from primary
    files).  This stage copies primary files from the directories
    where they are downloaded to, to the primary_files_dir, putting
    them into their proper locations.

    MAST downloads files into different locations than we use.

    When this stage finishes, there should be a primary_files_dir with
    the document and data files in it.

    During development, we are keeping the documents_dir and the
    mast_download_dir so we don't have to keep downloading the files,
    but in production, you could delete them.
    """

    def _copy_docs_files(
        self, bundle_segment: str, documents_dir: str, primary_files_dir: str
    ) -> None:
        if not os.path.isdir(documents_dir):
            raise ValueError(f"{documents_dir} doesn't exist.")
        try:
            PDS_LOGGER.open("Copy docs files to document directory")
            with make_osfs(documents_dir) as documents_fs, make_sv_osfs(
                primary_files_dir
            ) as primary_files_fs:
                new_dir_path = os.path.join(
                    to_segment_dir(bundle_segment),
                    to_segment_dir("document"),
                    to_segment_dir("phase2"),
                )
                primary_files_fs.makedirs(new_dir_path)
                for file in documents_fs.walk.files():
                    file_basename = os.path.basename(file)
                    new_file_path = os.path.join(new_dir_path, file_basename)
                    PDS_LOGGER.log("info", f"Copy {file_basename} to {new_file_path}")
                    fs.copy.copy_file(
                        documents_fs, file, primary_files_fs, new_file_path
                    )
        except Exception as e:
            PDS_LOGGER.error(e)
        finally:
            PDS_LOGGER.close()
        # shutil.rmtree(documents_dir)
        # assert not os.path.isdir(documents_dir)

    def _copy_fits_files(
        self, bundle_segment: str, mast_downloads_dir: str, primary_files_dir: str
    ) -> None:
        if not os.path.isdir(mast_downloads_dir):
            raise ValueError(f"{mast_downloads_dir} doesn't exist.")
        try:
            PDS_LOGGER.open("Copy fits files to corresponding directories")
            with make_osfs(mast_downloads_dir) as mast_downloads_fs, make_sv_osfs(
                primary_files_dir
            ) as primary_files_fs:

                # Walk the mast_downloads_dir for FITS file and file
                # them into the COW filesystem.
                for filepath in mast_downloads_fs.walk.files(filter=["*.fits"]):
                    parts = fs.path.iteratepath(filepath)
                    depth = len(parts)
                    if depth != 3:
                        raise ValueError(f"{parts} length is not 3.")
                    # New way: product name comes from the filename
                    _, _, filename = parts
                    filename = filename.lower()
                    hst_filename = HstFilename(filename)
                    product = hst_filename.rootname()
                    instrument_name = hst_filename.instrument_name()
                    suffix = hst_filename.suffix()

                    collection_type = get_collection_type(
                        suffix=suffix, instrument_id=instrument_name
                    )
                    coll = f"{collection_type}_{instrument_name}_{suffix}"

                    new_path = fs.path.join(
                        to_segment_dir(bundle_segment),
                        to_segment_dir(coll),
                        to_segment_dir(product),
                        filename,
                    )
                    dirs, filename = fs.path.split(new_path)
                    primary_files_fs.makedirs(dirs)
                    PDS_LOGGER.log("info", f"Copy {filename} to {new_path}")
                    fs.copy.copy_file(
                        mast_downloads_fs, filepath, primary_files_fs, new_path
                    )

            if not os.path.isdir(primary_files_dir + "-sv"):
                raise ValueError(f"{primary_files_dir + '-sv'} doesn't exist.")
            # # If I made it to here, it should be safe to delete the downloads
            # shutil.rmtree(mast_downloads_dir)
            # assert not os.path.isdir(mast_downloads_dir)
        except Exception as e:
            PDS_LOGGER.error(e)
        finally:
            PDS_LOGGER.close()

    def _run(self) -> None:
        documents_dir: str = self.documents_dir()
        mast_downloads_dir: str = self.mast_downloads_dir()
        primary_files_dir: str = self.primary_files_dir()

        if os.path.isdir(self.deliverable_dir()):
            raise ValueError(
                f"{self.deliverable_dir()} cannot exist for CopyPrimaryFiles."
            )

        if not self._bundle_segment.startswith("hst_"):
            raise ValueError(f"{self._bundle_segment} doesn't start with 'hst'.")
        if not self._bundle_segment[-5:].isdigit():
            raise ValueError(f"{self._bundle_segment} doesn't end with 5 digits.")

        self._copy_docs_files(self._bundle_segment, documents_dir, primary_files_dir)
        self._copy_fits_files(
            self._bundle_segment, mast_downloads_dir, primary_files_dir
        )

        if not os.path.isdir(primary_files_dir + "-sv"):
            raise ValueError(f"{primary_files_dir + '-sv'} doesn't exist.")
