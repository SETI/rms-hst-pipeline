import time
from typing import Any, Callable, Dict, List, Tuple

import julian
from astropy.table import Table
from astropy.table.row import Row
from requests.exceptions import ConnectionError

Julian = float


def filter_table(row_predicate: Callable[[Row], bool], table: Table) -> Table:
    to_delete = [n for (n, row) in enumerate(table) if not row_predicate(row)]
    copy = table.copy()
    copy.remove_rows(to_delete)
    return copy


def ymd_to_mjd(y: int, m: int, d: int) -> Julian:
    days = julian.day_from_ymd(y, m, d)
    return julian.mjd_from_day(days)


def ymd_tuple_to_mjd(ymd: Tuple[int, int, int]) -> Julian:
    y, m, d = ymd
    return ymd_to_mjd(y, m, d)


def ymdhms_format_from_mjd(mjd: float) -> str:
    (d, s) = julian.day_sec_from_mjd(mjd)
    return julian.ymdhms_format_from_day_sec(d, s)


def get_table_with_retries(mast_call: Callable[[], Table], max_retries: int) -> Table:
    retry = 0
    for retry in range(max_retries):
        try:
            table = mast_call()
            return table
        except ConnectionError as e:
            retry = retry + 1
            print(f"retry #{retry}: {e}")
            time.sleep(1)
    raise RuntimeError("get_table_with_retries() timed out")


############################################################


def now_mjd() -> float:
    return julian.mjd_from_time(time.time())


def mjd_range_to_now(last_check_mjd: float) -> List[float]:
    return [last_check_mjd, now_mjd()]


def table_to_list_of_dicts(table: Table) -> List[Dict[str, Any]]:
    """
    A utility function: the tables returned by astroquery are too
    large to be human-readable.
    """
    table_len = len(table)

    def mk_dict(i: int) -> Dict[str, Any]:
        return {key: table[key][i] for key in table.colnames}

    return [mk_dict(i) for i in range(table_len)]
