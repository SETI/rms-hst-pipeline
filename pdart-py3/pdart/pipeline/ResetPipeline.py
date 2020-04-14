import os
import os.path
import shutil


def reset_pipeline(
    working_dir: str, documents_dir: str, mast_downloads_dir: str
) -> None:
    if not os.path.isdir(working_dir):
        return
    for entry in os.listdir(working_dir):
        fullpath = os.path.join(working_dir, entry)
        if fullpath not in [documents_dir, mast_downloads_dir]:
            if os.path.isdir(fullpath):
                shutil.rmtree(fullpath)
            else:
                os.unlink(fullpath)
