import abc

from pdart.pipeline.Directories import Directories


class Stage(metaclass=abc.ABCMeta):
    def __init__(
        self, bundle_segment: str, dirs: Directories, proposal_id: int
    ) -> None:
        self.bundle_segment = bundle_segment
        self.dirs = dirs
        self.proposal_id = proposal_id

    def __call__(self) -> None:
        self._run()

    @abc.abstractmethod
    def _run(self) -> None:
        pass
