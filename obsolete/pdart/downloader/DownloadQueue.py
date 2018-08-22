from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime


def get_last_check_datetime():
    # type: () -> datetime.datetime
    assert False, 'unimplemented'


def set_last_check_datetime(last):
    # type: (datetime.datetime) -> None
    assert False, 'unimplemented'


def get_utc_time_now():
    # type: () -> datetime.datetime
    return datetime.datetime.utcnow()


def get_changed_bundles(last_check_datetime):
    # type: (datetime.datetime) -> List[str]
    assert False, 'unimplemented'


def update_bundle_ids(new_bundle_ids):
    # type: (List[str]) -> None
    ids = get_bundle_ids()
    ids.update(new_bundle_ids)
    set_bundle_ids(ids)


def get_bundle_ids():
    # type: () -> Set[str]
    assert False, 'unimplemented'


def set_bundle_ids(bundle_ids):
    # type: (Set[str]) -> None
    assert False, 'unimplemented'


def update_download_queue():
    # type: () -> None
    last = get_last_check_datetime()
    now = get_utc_time_now()
    # There's a race condition right here: new changes could creep in.
    # But that only means they'll get enqueued a second time, since
    # their change date is after 'now'.  Or I could ask for a range
    # bounded by the variable "now" (not the actual now on the HST
    # server).
    bundle_ids = get_changed_bundles(last)
    update_bundle_ids(bundle_ids)
    set_last_check_datetime(now)
