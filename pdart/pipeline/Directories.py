import os.path

class Directories(object):
    """
    We use this object as a way to configure where to find data files.
    This allows us to move the archives around or even split them over
    multiple hard disks in the future.
    """
    def __init__(self, base_dirpath):
        # type: (unicode) -> None
        self.base_dirpath = base_dirpath

    def working_dir(self, proposal_id):
        # type: (int) -> unicode
        hst_segment = 'hst_%05d' % proposal_id
        return os.path.join(self.base_dirpath, hst_segment)

    def mast_downloads_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id), 'mastDownload')

    def next_version_deltas_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id),
                            'next-version-deltas-sv')

    def archive_next_version_fits_and_docs_deltas_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id),
                            'archive-next-version-fits-and-docs-deltas-mv')

    def archive_next_version_browse_deltas_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id),
                            'archive-next-version-browse-deltas-sv')

    def archive_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id),
                            'archive-mv')

    def new_versions_dir(self, proposal_id):
        return os.path.join(self.working_dir(proposal_id),
                            'last-version-sv')

