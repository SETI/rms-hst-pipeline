import abc
import os.path

# TODO Make this abstract and implement for development and also for
# big-data.


def make_directories() -> "Directories":
    import socket

    hostname = socket.gethostname()
    if hostname == "Marks-iMac.local":
        return ProductionDirectories("/Volumes/AKBAR/working-dir")
    else:
        return DevDirectories("tmp-working-dir")


class Directories(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def working_dir(self, proposal_id: int) -> str:
        pass

    def mast_downloads_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "mastDownload")

    def next_version_deltas_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "next-version")

    def primary_files_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "primary-files")

    def documents_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "documentDownload")

    def archive_primary_deltas_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "primary")

    def archive_browse_deltas_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "browse")

    def archive_label_deltas_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "label")

    def archive_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "archive")

    def deliverable_dir(self, proposal_id: int) -> str:
        return os.path.join(
            self.working_dir(proposal_id), f"hst_{proposal_id:05}-deliverable"
        )


class DevDirectories(Directories):
    """
    For Nedervold use developing on own machine.
    """

    def __init__(self, base_dirpath: str) -> None:
        self.base_dirpath = base_dirpath

    def working_dir(self, proposal_id: int) -> str:
        hst_segment = f"hst_{proposal_id:05}"
        return os.path.join(self.base_dirpath, hst_segment)


class ProductionDirectories(Directories):
    """
    For production use running on Mark's IMac.
    """

    def __init__(self, base_dirpath: str) -> None:
        self.base_dirpath = base_dirpath

    def working_dir(self, proposal_id: int) -> str:
        hst_segment = f"hst_{proposal_id:05}"
        return os.path.join(self.base_dirpath, hst_segment)
