from typing import TYPE_CHECKING
import os.path

from pdart.pipeline.Utils import *

if TYPE_CHECKING:
    pass

def make_browse_files(archive_dir,
                      archive_next_version_fits_and_docs_deltas_dir,
                      archive_next_version_browse_deltas_dir):
    # type: (unicode, unicode, unicode) -> None
    assert os.path.isdir(archive_dir)
    assert os.path.isdir(archive_next_version_fits_and_docs_deltas_dir)

    assert False
    
