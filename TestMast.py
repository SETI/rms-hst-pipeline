import csv
import urllib

_DATA_SET = 'hst'

_QUERY_DEFAULTS = {
    'sci_instrume': 'ACS,WFPC2,WFC3',
    'sci_mtflag': 'T',
    'outputformat': 'CSV',
    'resolver': "don'tresolve"
    }


def _make_url(query_dict):
    params = urllib.urlencode(query_dict)
    url = 'https://archive.stsci.edu/%s/search.php?action=Search&%s' % \
        (_DATA_SET, params)
    return url


def query_dicts(query_args):
    """
    Given a dictionary of HST search arguments, return a generator of
    dictionaries, each dictionary containing a single result.
    """
    # copy the dict so we get a fresh copy
    query_dict = dict(_QUERY_DEFAULTS)
    query_dict.update(query_args)
    url = _make_url(query_dict)
    (filename, headers) = \
        urllib.urlretrieve(url, '/Users/spaceman/pdart/mast.result.txt')
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        # Skip the row containing field types
        reader.next()
        for row in reader:
            yield row


if __name__ == '__main__':
    query_args = {
        'max_records': '1'
        }
    ds = query_dicts(query_args)
    for d in ds:
        print d
        break
