"""
Functionality to retrieve HST data from MAST.
"""
import urllib

_DATA_SET = 'hst'
# type: str

_QUERY_DEFAULTS = {
    'sci_instrume': 'ACS,WFPC2,WFC3',
    'sci_mtflag': 'T',
    'outputformat': 'CSV',
    'resolver': "don'tresolve"
    }
# type: Dict[str, str]


def _make_query_dict(args):
    # type: (Dict) -> Dict
    """
    Given a dictionary of query arguments for an HST search, combine
    it with default values and return that dictionary.
    """
    res = dict(_QUERY_DEFAULTS)
    res.update(args)
    return res


def _make_url(query_dict):
    # type: (Dict) -> str
    """
    Given a dictionary of search parameters, return the URL to perform
    the search.
    """
    params = urllib.urlencode(query_dict)
    url = 'https://archive.stsci.edu/%s/search.php?action=Search&%s' % \
        (_DATA_SET, params)
    return url


def make_url(**kwargs):
    # type: (...) -> str
    """
    Given a list of search parameters, return the URL to perform the
    search.
    """
    return _make_url(_make_query_dict(kwargs))


def hst_search(**kwargs):
    # type: (...) -> str
    """
    Given a list of search parameters, perform an HST search and
    return a filename containing the search results.
    """
    query_dict = _make_query_dict(kwargs)
    url = _make_url(query_dict)
    return urllib.urlretrieve(url)[0]
