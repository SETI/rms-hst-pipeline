import time

import pdart.add_pds_tools
import julian
from requests.exceptions import ConnectionError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.table import Table
    from typing import Any, Callable, Dict, List
    Julian = float


def filter_table(row_predicate, table):
    # type: (Callable[[Table], bool], Table) -> Table
    to_delete = [n for (n, row) in enumerate(table) if row_predicate(row)]
    copy = table.copy()
    copy.remove_rows(to_delete)
    return copy


def ymd_to_mjd(y, m, d):
    # type: (int, int, int) -> Julian
    days = julian.day_from_ymd(y, m, d)
    return julian.mjd_from_day(days)


def ymdhms_format_from_mjd(mjd):
    # type: (float) -> str
    (d, s) = julian.day_sec_from_mjd(mjd)
    return julian.ymdhms_format_from_day_sec(d, s)


def get_table_with_retries(mast_call, msg):
    # type: (Callable[[], None], str) -> Table
    retry = 0
    table = None
    while table is None:
        try:
            table = mast_call()
        except ConnectionError:
            retry = retry + 1
            print 'Retry #%d: %s' % (retry, msg)
            time.sleep(1)
    return table

############################################################


def now_mjd():
    # type: () -> float
    return julian.mjd_from_time(time.time())


def mjd_range_to_now(last_check_mjd):
    # type: (float) -> List[float]
    return [last_check_mjd, now_mjd()]


def table_to_list_of_dicts(table):
    # type: (Table) -> List[Dict[str, Any]]
    """
    A utility function: the tables returned by astroquery are too
    large to be human-readable.
    """
    table_len = len(table)

    def mk_dict(i):
        # type: (int) -> Dict[str, Any]
        return {key: table[key][i] for key in table.colnames}

    return [mk_dict(i) for i in range(table_len)]
