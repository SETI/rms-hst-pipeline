import abc

from pdart.pipeline.Directories import Directories


class Stage(metaclass=abc.ABCMeta):
    def __init__(
        self, bundle_segment: str, dirs: Directories, proposal_id: int
    ) -> None:
        self._bundle_segment = bundle_segment
        self._dirs = dirs
        self._proposal_id = proposal_id

    def working_dir(self) -> str:
        return self._dirs.working_dir(self._proposal_id)

    def mast_downloads_dir(self) -> str:
        return self._dirs.mast_downloads_dir(self._proposal_id)

    def primary_files_dir(self) -> str:
        return self._dirs.primary_files_dir(self._proposal_id)

    def documents_dir(self) -> str:
        return self._dirs.documents_dir(self._proposal_id)

    def archive_primary_deltas_dir(self) -> str:
        return self._dirs.archive_primary_deltas_dir(self._proposal_id)

    def archive_browse_deltas_dir(self) -> str:
        return self._dirs.archive_browse_deltas_dir(self._proposal_id)

    def archive_label_deltas_dir(self) -> str:
        return self._dirs.archive_label_deltas_dir(self._proposal_id)

    def archive_dir(self) -> str:
        return self._dirs.archive_dir(self._proposal_id)

    def deliverable_dir(self) -> str:
        return self._dirs.deliverable_dir(self._proposal_id)

    ##############################

    def __call__(self) -> None:
        self._run()

    @abc.abstractmethod
    def _run(self) -> None:
        pass
