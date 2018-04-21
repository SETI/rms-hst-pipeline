# SCRIPT: A script to estimate how often files change.

import sys
import time

import pdart.add_pds_tools
import julian
from astroquery.mast import Observations
from requests.exceptions import ConnectionError


def ymdhms_format_from_mjd(mjd):
    # type: (int) -> str
    (d, s) = julian.day_sec_from_mjd(mjd)
    return julian.ymdhms_format_from_day_sec(d, s)


def ymd_to_mjd(y, m, d):
    # type: (int, int, int) -> float
    days = julian.day_from_ymd(y, m, d)
    return julian.mjd_from_day(days)


def get_observations_table(startDate, endDate):
    table = None
    while table is None:
        try:
            sys.stdout.flush()
            table = Observations.query_criteria(
                dataproduct_type=['image'],
                dataRights='PUBLIC',
                obs_collection=['HST'],
                t_obs_release=(startDate, endDate),
                mtFlag=True)
        except ConnectionError:
            print >> sys.stderr, \
                '@#$%', \
                ymdhms_format_from_mjd(startDate), \
                'Connection failed; retrying.'
            time.sleep(1)
    return table


def get_products_table(table):
    return Observations.get_product_list(table)


def run_test():
    today = int(ymd_to_mjd(2018, 3, 26))
    start_test = today - 365
    for d in xrange(start_test, today - 1):
        try:
            table = get_observations_table(d, d + 1)
            table = get_products_table(table)
            proposals = sorted(list(set(table['proposal_id'])))
            for proposal in proposals:
                recs = [rec for rec in table if rec['proposal_id'] == proposal]
                print '****', \
                    ymdhms_format_from_mjd(d), \
                    ('hst_%s changed; has %d file(s)' % (proposal, len(recs)))
                for fn in sorted(rec['productFilename'] for rec in recs):
                    print '  ', fn
        except KeyError:
            print '----', ymdhms_format_from_mjd(d), 'Nothing found.'
        sys.stdout.flush()


if __name__ == '__main__':
    run_test()
