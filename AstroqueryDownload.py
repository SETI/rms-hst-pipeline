import sys

from pdart.astroquery.Astroquery import MastSlice


def sample(slice):
    def to_gigabytes(bytes):
        return bytes / 1024.0 / 1024 / 1024

    proposal_ids = slice.get_proposal_ids()
    for proposal_id in proposal_ids:
        product_set = slice.to_product_set(proposal_id)
        sz = to_gigabytes(product_set.download_size())
        cnt = product_set.product_count()
        if cnt > 0:
            print "%.1f GB in %d products from id %d" % (sz, cnt, proposal_id)
            sys.stdout.flush()


def run():
    # type: () -> None
    today = (2018, 3, 26)
    start = (1900, 1, 1)
    slice = MastSlice(start, today)
    if False:
        slice.download_products_by_id(7240, './tmp')
    else:
        sample(slice)


if __name__ == '__main__':
    run()
