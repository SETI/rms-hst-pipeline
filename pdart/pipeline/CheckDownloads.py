import os
import os.path
from typing import TYPE_CHECKING

from pdart.astroquery.Astroquery import MastSlice

def check_downloads(working_dir, proposal_id):
    # type: (unicode, int) -> None

    # first pass, <working_dir> shouldn't exist; second pass
    # <working_dir>/mastDownload should not exist.
    assert not os.path.isdir(os.path.join(working_dir, 'mastDownload'))

    # TODO These dates are wrong.  Do I need to do some optimization
    # here?
    slice = MastSlice((1900, 1, 1), (2025, 1, 1), proposal_id)
    proposal_ids = slice.get_proposal_ids()
    assert proposal_id in proposal_ids, \
        "%d in %s" % (proposal_id, proposal_ids)
    product_set = slice.to_product_set(proposal_id)
    if not os.path.isdir(working_dir): 
        os.makedirs(working_dir)
    
    product_set.download(working_dir)
    

