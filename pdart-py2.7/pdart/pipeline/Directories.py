import abc
import os.path

# TODO Make this abstract and implement for development and also for
# big-data.

def make_directories():
    # type: () -> Directories
    import socket
    hostname = socket.gethostname()
    if hostname == 'Navin.local':
        return DevDirectories('tmp-working-dir')
    if hostname == 'Marks-iMac.local':
        return ProductionDirectories('/Volumes/AKBAR/working-dir')
    raise Exception('unknown hostname: ' + hostname)


class Directories(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def working_dir(self, proposal_id):
        # type: (int) -> unicode
        pass

    def mast_downloads_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "mastDownload")

    def next_version_deltas_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "next-version")

    def primary_files_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "primary-files")

    def documents_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "documentDownload")

    def archive_primary_deltas_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "primary")

    def archive_browse_deltas_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "browse")

    def archive_label_deltas_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "label")

    def archive_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), "archive")

    def deliverable_dir(self, proposal_id):
        return os.path.join(
            self.working_dir(proposal_id), "hst_%05d-deliverable" % proposal_id
        )

class DevDirectories(Directories):
    """
    For Nedervold use developing on own machine.
    """

    def __init__(self, base_dirpath):
        # type: (unicode) -> None
        self.base_dirpath = base_dirpath

    def working_dir(self, proposal_id):
        # type: (int) -> unicode
        hst_segment = "hst_%05d" % proposal_id
        return os.path.join(self.base_dirpath, hst_segment)

class ProductionDirectories(Directories):
    """
    For production use running on Mark's IMac.
    """
    def __init__(self, base_dirpath):
        # type: (unicode) -> None
        self.base_dirpath = base_dirpath

    def working_dir(self, proposal_id):
        # type: (int) -> unicode
        hst_segment = "hst_%05d" % proposal_id
        return os.path.join(self.base_dirpath, hst_segment)

