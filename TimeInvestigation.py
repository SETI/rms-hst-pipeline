import pprint
import re

import pyfits

import pdart.pds4.Archives

DATE_TIME_RE = re.compile('(DATE|TIME)', re.IGNORECASE)


def is_date_time(k):
    return DATE_TIME_RE.search(k) is not None


def search_for_datetime():
    archive = pdart.pds4.Archives.get_any_archive()
    for p in archive.products():
        filepath = p.absolute_filepath()
        try:
            fits = pyfits.open(filepath)
            try:
                h = fits[0].header
                dict = {}
                for k in h.keys():
                    if k != 'DATE' and is_date_time(k):
                        dict[k] = h[k]
                print '%s: %s' % (filepath, dict)
            finally:
                fits.close()
        except Exception as e:
            print '%s: Exception %s' % (filepath, e)


def search_for_datetime_patterns():
    archive = pdart.pds4.Archives.get_any_archive()
    patterns = {}
    for p in archive.products():
        suffix = p.collection().suffix()
        filepath = p.absolute_filepath()
        try:
            fits = pyfits.open(filepath)
            try:
                h = fits[0].header
                datetime_keys = frozenset([k for k in h.keys()
                                           if k != 'DATE' and is_date_time(k)])
                if suffix not in patterns:
                    patterns[suffix] = set()
                patterns[suffix].add(datetime_keys)
            finally:
                fits.close()
        except Exception as e:
            pass

    pprint.pprint(patterns)


if __name__ == '__main__':
    search_for_datetime_patterns()
