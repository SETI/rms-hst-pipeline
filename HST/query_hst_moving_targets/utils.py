##########################################################################################
# query_hst_moving_targets/utils.py
##########################################################################################

import julian

def ymd_tuple_to_mjd(ymd):
    """Return Modified Julian Date.
    Input:
        ymd:    a tuple of year, month, and day.
    """
    y, m, d = ymd
    days = julian.day_from_ymd(y, m, d)
    return julian.mjd_from_day(days)