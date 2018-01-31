from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime


def get_last_check_datetime():
    # type: () -> datetime.datetime
    assert False, 'unimplemented'


def set_last_check_datetime(last):
    # type: (datetime.datetime) -> None
    assert False, 'unimplemented'


def get_time_now():
    # type: () -> datetime.datetime
    assert False, 'unimplemented'


def get_changed_bundles(last_check_datetime):
    # type: (datetime.datetime) -> List[int]
    assert False, 'unimplemented'


def enqueue_proposal_ids(proposal_ids):
    # type: (List[int]) -> None
    assert False, 'unimplemented'


def update_download_queue():
    # type: () -> None
    last = get_last_check_datetime()
    now = get_time_now()
    # There's a race condition right here: new changes could creep in.
    # But that only means they'll get enqueued a second time, since
    # their change date is after 'now'.
    proposal_ids = get_changed_bundles(last)
    enqueue_proposal_ids(proposal_ids)
    set_last_check_datetime(now)
