from typing import TYPE_CHECKING
import pdart.add_pds_tools
import julian
import time

if TYPE_CHECKING:
    from astropy.table import Table
    from typing import Any, Dict, List


def ymdhms_format_from_mjd(mjd):
    # type: (float) -> str
    (d, s) = julian.day_sec_from_mjd(mjd)
    return julian.ymdhms_format_from_day_sec(d, s)


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
