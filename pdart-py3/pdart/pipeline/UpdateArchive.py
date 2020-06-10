import os
import os.path
import shutil

from pdart.fs.multiversioned.Multiversioned import Multiversioned, std_is_new
from pdart.pipeline.RecordChanges import CHANGES_DICT
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view


class UpdateArchive(MarkedStage):
    def _run(self) -> None:
        working_dir: str = self.working_dir()
        archive_dir: str = self.archive_dir()
        archive_primary_deltas_dir: str = self.archive_primary_deltas_dir()
        archive_browse_deltas_dir: str = self.archive_browse_deltas_dir()
        archive_label_deltas_dir: str = self.archive_label_deltas_dir()

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
        changes_dict_path = os.path.join(working_dir, CHANGES_DICT)
        os.remove(changes_dict_path)

        assert not os.path.isdir(archive_primary_deltas_dir + "-deltas-sv")
        assert not os.path.isdir(archive_browse_deltas_dir + "-deltas-sv")
        assert not os.path.isdir(archive_label_deltas_dir + "-deltas-sv")
        assert os.path.isdir(archive_dir)
        assert not os.path.isfile(changes_dict_path)
