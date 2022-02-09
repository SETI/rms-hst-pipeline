from typing import List
import abc
import os
import os.path

TWD = os.environ["TMP_WORKING_DIR"]

# TODO Make this abstract and implement for development and also for
# big-data.


def make_directories() -> "Directories":
    import socket

    hostname = socket.gethostname()
    if hostname == "Marks-iMac.local":
        return ProductionDirectories(
            ["/Volumes/AKBAR/working-dir", "/Volumes/PDART-8TB/working-dir"]
        )
    else:
        # Working path, it is pointing to external drive now
        return DevDirectories(TWD)


class Directories(object, metaclass=abc.ABCMeta):
    """
    An abstract class that provides a list of directories that stages
    can use.  By centralizing them, they can be edited in one place.
    If some part of the process uses too much disk space, part of it
    can be moved onto a different external disk just by editing this
    class.

    The default implementation though is to have all the directories
    live within the working_dir for the bundle.
    """

    @abc.abstractmethod
    def working_dir(self, proposal_id: int) -> str:
        pass

    def mast_downloads_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "mastDownload")

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
        bundle_segment = f"hst_{proposal_id:05}"
        return os.path.join(
            self.working_dir(proposal_id), f"{bundle_segment}-deliverable"
        )

    def deliverable_bundle_dir(self, proposal_id: int) -> str:
        bundle_segment = f"hst_{proposal_id:05}"
        return os.path.join(self.deliverable_dir(proposal_id), bundle_segment)

    def manifest_dir(self, proposal_id: int) -> str:
        return self.deliverable_dir(proposal_id)

    def validation_report_dir(self, proposal_id: int) -> str:
        return self.working_dir(proposal_id)

    def log_dir(self, proposal_id: int) -> str:
        return os.path.join(self.working_dir(proposal_id), "log")


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

    def __init__(self, base_dirpaths: List[str]) -> None:
        self.base_dirpaths = base_dirpaths

    def working_dir(self, proposal_id: int) -> str:
        hst_segment = f"hst_{proposal_id:05}"
        indx = proposal_id % len(self.base_dirpaths)
        return os.path.join(self.base_dirpaths[indx], hst_segment)
