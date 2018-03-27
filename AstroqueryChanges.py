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


def get_table(startDate, endDate):
    table = None
    while table is None:
        try:
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


def run_test():
    today = int(ymd_to_mjd(2018, 3, 26))
    start_test = today - 365
    for d in xrange(start_test, today - 1):
        try:
            table = get_table(d, d + 1)
            print '****', \
                ymdhms_format_from_mjd(d), \
                ('Found %d changed file(s):' % len(table))
            for rec in table:
                print ('  hst_%s' % rec['proposal_id']), \
                    rec['dataURL'], \
                    ymdhms_format_from_mjd(rec['t_obs_release'])
        except KeyError:
            print '----', ymdhms_format_from_mjd(d), 'Nothing found.'


if __name__ == '__main__':
    run_test()
