import urllib

_HST_BASE_URL = 'https://archive.stsci.edu/hst/search.php'


def hst_search(**kwargs):
    """
    Given a list of search parameters, perform an HST search and
    return a filename containing the search results.
    """
    url = '%s?action=Search&%s' % (_HST_BASE_URL, urllib.urlencode(kwargs))
    return urllib.urlretrieve(url)[0]
