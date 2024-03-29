from typing import Any, Optional, Tuple

def day_from_iso(strings: str, validate: bool = True, strop: bool = False) -> float: ...
def day_from_ymd(y: int, m: int, d: int) -> int: ...
def day_sec_from_mjd(jd: float) -> Tuple[int, int]: ...
def iso_from_tai(
    tai: float, digits: Optional[int] = None, ymd: bool = True, suffix: str = ""
) -> str: ...
def mjd_from_day(d: int) -> int: ...
def mjd_from_day_sec(day: float, sec: float) -> int: ...
def mjd_from_time(time: float) -> float: ...
def sec_from_iso(strings: str, validate: bool = True, strop: bool = False) -> float: ...
def tai_from_iso(iso: str, validate: bool = True, strip: bool = False) -> float: ...
def ymdhms_format_from_day_sec(
    day: int,
    sec: int,
    sep: str = "T",
    digits: Optional[int] = None,
    suffix: str = "",
    buffer: Optional[Any] = None,
) -> str: ...
