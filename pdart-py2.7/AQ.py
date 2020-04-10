from pdart.astroquery.Astroquery import MastSlice

# NOTE: I should add a test to verify column names for both
# observations and products.  I should make it a utility function of
# MastSlice and ProductSet.

def aq():
    today = (2019, 11, 1)
    start = (2010, 1, 1)
    slice = MastSlice(start, today)
    # print(sorted(slice.observations_table[0].colnames))
    # return
    values = set()
    for proposal_id in slice.get_proposal_ids():
        print ('Proposal hst_%05d:' % proposal_id)
        product_set = slice.to_product_set(proposal_id)
        for product in product_set.table:
            # print ('    %s' % product['productFilename'])
            prvversion = product['prvversion']
            print ('    %s: %r' %
                   (product['productFilename'], prvversion))
            values.add(repr(prvversion))
    print 'All prvversion values used from %s to %s' % (start, today)
    print sorted(list(values))

if __name__ == '__main__':
    aq()
