"""
This module is a script to explore querying MAST.
"""
from pdart.hst.HstSearch import *

# TODO How do I limit on the archive date? sci_archive_date
# TODO How do I get LIDs for the files? sci_data_set_name
# TODO How do I get the files?  st_dads is an asynch service.


def run():
    # type: () -> None
    print 'Hello, from SearchDev!'
    if False:
        columns = 'sci_archive_date,sci_data_set_name,' + \
            'sci_targname,sci_target_descrip'
        url = make_url(max_records=100,
                       sci_archive_date='>=2016-07-01',
                       selectedColumnsCsv=columns
                       )
    else:
        url = make_url(max_records=200000,
                       selectedColumnsCsv='sci_archive_date,sci_data_set_name')
    print 'url would be:'
    print url
    fp = urllib.urlretrieve(url, '/Users/spaceman/pdart/mast.result.txt')[0]
    with open(fp, 'r') as f:
        for l in f:
            print l,

if __name__ == '__main__':
    run()
