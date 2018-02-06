import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fs.base import FS
    Download = int


def get_last_update_datetime():
    # type: () -> datetime.datetime
    assert False, 'unimplemented'


def get_utc_time_now():
    # type: () -> datetime.datetime
    return datetime.datetime.utcnow()


def downloads_needed(proposal_id, start_time, end_time):
    # type: (int, datetime.datetime, datetime.datetime) -> List[Download]
    assert False, 'unimplemented'


def download_file(proposal_id, download, download_fs):
    # type: (int, Download, FS) -> None
    assert False, 'unimplemented'


def update_bundle(download_fs, cow_fs):
    # type: (FS, FS) -> None
    assert False, 'unimplemented'


def set_last_update_datetime(last):
    # type: (datetime.datetime) -> None
    assert False, 'unimplemented'


def download_bundle_changes(proposal_id, download_fs, cow_fs):
    # type: (int, FS, FS) -> None

    last = get_last_update_datetime()
    while True:
        now = get_utc_time_now()
        downloads = downloads_needed(proposal_id, last, now)
        if downloads:
            download_file(proposal_id, downloads[0], download_fs)
        else:
            break
    update_bundle(download_fs, cow_fs)
    set_last_update_datetime(now)