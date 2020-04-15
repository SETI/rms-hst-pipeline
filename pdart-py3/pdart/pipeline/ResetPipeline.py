import os
import os.path
import shutil
from pdart.pipeline.Stage import Stage


class ResetPipeline(Stage):
    def _run(self) -> None:
        working_dir: str = self.dirs.working_dir(self.proposal_id)
        documents_dir: str = self.dirs.documents_dir(self.proposal_id)
        mast_downloads_dir: str = self.dirs.mast_downloads_dir(self.proposal_id)

        if not os.path.isdir(working_dir):
            return
        for entry in os.listdir(working_dir):
            fullpath = os.path.join(working_dir, entry)
            if fullpath not in [documents_dir, mast_downloads_dir]:
                if os.path.isdir(fullpath):
                    shutil.rmtree(fullpath)
                else:
                    os.unlink(fullpath)
