import os
import os.path
from typing import Optional

from astropy.table.row import Row

from pdart.astroquery.Astroquery import MastSlice, ProductSet, filter_table
from pdart.pipeline.Stage import MarkedStage

from pdart.logging import PDS_LOGGER


class CheckDownloads(MarkedStage):
    """
    This stage downloads all the data files for this bundle
    (equivalently, proposal_id) into the mast_downloads_dir, creating
    it if possible.

    This is called CheckDownloads instead of DownloadData because in
    the future we would like to first check what has changed, and then
    download only those files that have changed.

    Currently, we have nothing implemented to do that, so we just
    download everything and then check the contents of the files.

    After this stage runs, there should be a mast_downloads_dir and it
    should contain some datafiles.
    """

    def _do_downloads(
        self,
        working_dir: str,
        mast_downloads_dir: str,
        proposal_id: int,
    ) -> None:
        PDS_LOGGER.open("Download datafiles")
        # first pass, <working_dir> shouldn't exist; second pass
        # <working_dir>/mastDownload should not exist.
        if os.path.isdir(mast_downloads_dir):
            raise ValueError("<working_dir>/mastDownload should not exist.")

        # TODO These dates are wrong; they potentially collect too
        # much.  Do I need to reduce the range of dates here?
        slice = MastSlice((1900, 1, 1), (2025, 1, 1), proposal_id)
        proposal_ids = slice.get_proposal_ids()
        if proposal_id not in proposal_ids:
            raise KeyError(f"{proposal_id} not in {proposal_ids}")
        # get files from full list of ACCEPTED_SUFFIXES
        product_set = slice.to_product_set(proposal_id)
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir)

        # TODO I should also download the documents here.
        product_set.download(working_dir)

        # TODO This might fail if there are no files.  Which might not be
        # a bad thing.
        PDS_LOGGER.log("info", f"::::::::::mast_downloads_dir: {mast_downloads_dir}")
        PDS_LOGGER.log("info", f"Download datafiles to {mast_downloads_dir}")
        if not os.path.isdir(mast_downloads_dir):
            raise ValueError(f"{mast_downloads_dir} doesn't exist.")
        PDS_LOGGER.close()

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        mast_downloads_dir: str = self.mast_downloads_dir()

        if os.path.isdir(self.deliverable_dir()):
            raise ValueError(
                f"{self.deliverable_dir()} cannot exist for CheckDownloads."
            )

        if not os.path.isdir(mast_downloads_dir):
            self._do_downloads(
                working_dir,
                mast_downloads_dir,
                self._proposal_id,
            )
