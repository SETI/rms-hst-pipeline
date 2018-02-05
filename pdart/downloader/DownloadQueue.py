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
    # type: (datetime.datetime) -> List[int]
    assert False, 'unimplemented'


def update_proposal_ids(new_proposal_ids):
    # type: (List[int]) -> None
    ids = get_proposal_ids()
    ids.update(new_proposal_ids)
    set_proposal_ids(ids)


def get_proposal_ids():
    # type: () -> Set[int]
    assert False, 'unimplemented'


def set_proposal_ids(proposal_ids):
    # type: (Set[int]) -> None
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
    proposal_ids = get_changed_bundles(last)
    update_proposal_ids(proposal_ids)
    set_last_check_datetime(now)
