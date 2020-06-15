import os
import os.path
import shutil
from pdart.pipeline.Stage import MarkedStage


class ResetPipeline(MarkedStage):
    def _run(self) -> None:
        working_dir: str = self.working_dir()
        documents_dir: str = self.documents_dir()
        mast_downloads_dir: str = self.mast_downloads_dir()

        if not os.path.isdir(working_dir):
            return
        for entry in os.listdir(working_dir):
            fullpath = os.path.join(working_dir, entry)
            if fullpath not in [documents_dir, mast_downloads_dir]:
                if os.path.isdir(fullpath):
                    shutil.rmtree(fullpath)
                else:
                    os.unlink(fullpath)
