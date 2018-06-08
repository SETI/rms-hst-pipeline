import os
import os.path

from pdart.astroquery.Astroquery import MastSlice


def mkdirs(dir):
    # type: (unicode) -> None
    if not os.path.exists(dir):
        os.makedirs(dir)


def bulk_download(dl_root_dir):
    # type: (unicode) -> None
    mkdirs(dl_root_dir)
    today = (2018, 12, 31)
    start = (1900, 1, 1)
    slice = MastSlice(start, today)
    proposal_ids = slice.get_proposal_ids()
    for proposal_id in proposal_ids:
        product_set = slice.to_product_set(proposal_id)
        if product_set.product_count() > 0:
            hst_code = 'hst_%05d' % (proposal_id,)
            size = product_set.download_size()
            size_in_megs = size / 1024.0 / 1024
            print 'PROPOSAL %s %.2f MB' % (hst_code, size_in_megs)
            dl_dir = os.path.join(dl_root_dir, hst_code)
            mkdirs(dl_dir)
            product_set.download(dl_dir)


if __name__ == '__main__':
    bulk_download('tmp_bd')
