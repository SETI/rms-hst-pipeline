import datetime
import os
import os.path
import time
from typing import TYPE_CHECKING

from pdart.astroquery.Astroquery import MastSlice

if TYPE_CHECKING:
    from typing import Any, Callable


def mkdirs(dir):
    # type: (unicode) -> None
    if not os.path.exists(dir):
        os.makedirs(dir)


_SECOND = 1
_MINUTE = 60 * _SECOND
_TEN_MINUTES = 10 * _MINUTE


def do_with_retries(msg, action):
    # type: (str, Callable[[], Any]) -> Any
    done = False
    retries = 0
    while not done:
        try:
            return action()
        except IOError as e:
            retries = retries + 1
            if retries > 10:
                raise e
            else:
                print ('%s: sleeping ten minutes (retry #%d)...' %
                       (msg, retries))
                time.sleep(_TEN_MINUTES)


def bulk_download(dl_root_dir):
    # type: (unicode) -> None
    mkdirs(dl_root_dir)
    today = (2019, 12, 31)
    start = (1900, 1, 1)
    slice = MastSlice(start, today)
    proposal_ids = slice.get_proposal_ids()
    # Continuing yesterday's bulk download
    if True:
        # proposal_ids = [id for id in proposal_ids if id >= 14092]
        proposal_ids = [11187]
    else:
        # sampling: let's take five from the already downloaded list
        proposal_ids = [id for id in proposal_ids if id <= 14092]
        len_ = len(proposal_ids)
        l7 = len_ / 7
        selected_ids = [proposal_ids[i * l7] for i in range(1, 6)]
        proposal_ids = selected_ids

    for proposal_id in proposal_ids:
        hst_code = 'hst_%05d' % (proposal_id,)

        msg = "get product set for " + hst_code

        def get_product_set_action():
            return slice.to_product_set(proposal_id)
        product_set = do_with_retries(msg, get_product_set_action)

        if product_set.product_count() > 0:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            count = product_set.product_count()
            size = product_set.download_size()
            size_in_megs = size / 1024.0 / 1024

            print '%s: %s has %d products in %.2f MB' % (
                now, hst_code, count, size_in_megs)
            dl_dir = os.path.join(dl_root_dir, hst_code)
            mkdirs(dl_dir)

            def download_action():
                product_set.download(dl_dir)

            msg2 = "downloading " + hst_code
            do_with_retries(msg2, download_action)


if __name__ == '__main__':
    # bulk_download('/Volumes/PDART-5TB Part Deux/bulk-download')
    bulk_download('/Users/spaceman/pdart/new-bulk-download')
