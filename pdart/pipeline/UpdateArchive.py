import os
import os.path
import shutil

from pdart.fs.multiversioned.Multiversioned import Multiversioned, std_is_new
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.utils import make_osfs, make_sv_deltas, make_version_view


class UpdateArchive(MarkedStage):
    """
    Insert all new files into the archive as a new version.
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        archive_dir: str = self.archive_dir()
        archive_primary_deltas_dir: str = self.archive_primary_deltas_dir()
        archive_browse_deltas_dir: str = self.archive_browse_deltas_dir()
        archive_label_deltas_dir: str = self.archive_label_deltas_dir()

        if os.path.isdir(self.deliverable_dir()):
            raise ValueError(
                f"{self.deliverable_dir()} cannot exist for UpdateArchive."
            )

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas, make_sv_deltas(
            sv_deltas, archive_browse_deltas_dir
        ) as browse_deltas, make_sv_deltas(
            browse_deltas, archive_label_deltas_dir
        ) as label_deltas:

            # TODO I *think* this is a hack and will only work for the
            # initial import...but maybe I accidentally wrote better code
            # than I think and it'll work for all cases.  Investigate.
            mv = Multiversioned(archive_osfs)
            mv.update_from_single_version(std_is_new, label_deltas)

        shutil.rmtree(archive_primary_deltas_dir + "-deltas-sv")
        shutil.rmtree(archive_browse_deltas_dir + "-deltas-sv")
        shutil.rmtree(archive_label_deltas_dir + "-deltas-sv")

        if os.path.isdir(archive_primary_deltas_dir + "-deltas-sv"):
            raise ValueError(f"{archive_primary_deltas_dir}-deltas-sv shouldn't exist.")
        if os.path.isdir(archive_browse_deltas_dir + "-deltas-sv"):
            raise ValueError(f"{archive_browse_deltas_dir}-deltas-sv shouldn't exist.")
        if os.path.isdir(archive_label_deltas_dir + "-deltas-sv"):
            raise ValueError(f"{archive_label_deltas_dir}-deltas-sv shouldn't exist.")
        if not os.path.isdir(archive_dir):
            raise ValueError(f"{archive_dir} doesn't exist.")
