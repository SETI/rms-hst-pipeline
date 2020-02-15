from typing import TYPE_CHECKING
import os.path

from pdart.pipeline.Utils import *

if TYPE_CHECKING:
    pass

def make_browse_files(archive_dir,
                      fits_and_docs_deltas_dir,
                      browse_deltas_dir):
    # type: (unicode, unicode, unicode) -> None
    assert os.path.isdir(archive_dir + '-mv')
    assert os.path.isdir(fits_and_docs_deltas_dir + '-deltas-mv')

    assert False
    
