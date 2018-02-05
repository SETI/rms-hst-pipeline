import datetime

from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker

from pdart.downloader.SqlAlchTables import *

if TYPE_CHECKING:
    from typing import Set, Tuple


def _string_to_int_set(str):
    # type: (str) -> Set[int]
    return set([int(s) for s in str.split()])


def _int_set_to_string(intset):
    # type: (Set[int]) -> str
    return ' '.join([str(n) for n in sorted(list(intset))])


def create_downloader_db_from_os_filepath(os_filepath):
    # type: (unicode) -> DownloaderDB
    return DownloaderDB('sqlite:///' + os_filepath)


def create_downloader_db_in_memory():
    # type: () -> DownloaderDB
    return DownloaderDB('sqlite:///')


class DownloaderDB(object):
    def __init__(self, url):
        # type: (unicode) -> None
        self.url = url
        self.engine = create_engine(url)
        self.session = sessionmaker(bind=self.engine)()

    def create_tables(self):
        # type: () -> None
        create_tables(self.engine)

    def get_last_update_datetime(self, prop_id):
        # type: (int) -> datetime.datetime
        return self.session.query(
            UpdateDatetime).filter(
            UpdateDatetime.proposal_id == prop_id).one().update_datetime

    def set_last_update_datetime(self, prop_id, last):
        # type: (int, datetime.datetime) -> None
        if self.session.query(
                exists().where(
                    UpdateDatetime.proposal_id == prop_id)).scalar():
            # No need for locking the row here since I'm not modifying
            # the value, just replacing it.
            self.session.query(
                UpdateDatetime).update({'proposal_id': prop_id,
                                        'update_datetime': last})
        else:
            self.session.add(UpdateDatetime(proposal_id=prop_id,
                                            update_datetime=last))

    def get_last_check(self):
        # type: () -> Tuple[datetime.datetime, Set[int]]
        check = self.session.query(
            CheckDatetime).one()
        return (check.check_datetime, _string_to_int_set(check.proposal_ids))

    def set_last_check(self, last, prop_id_set):
        # type: (datetime.datetime, Set[int]) -> None
        prop_ids = _int_set_to_string(prop_id_set)
        if self.session.query(CheckDatetime).count():
            # No need for locking the row here since I'm not modifying
            # the value, just replacing it.
            self.session.query(
                CheckDatetime).update({'check_datetime': last,
                                       'proposal_ids': prop_ids})
        else:
            self.session.add(CheckDatetime(check_datetime=last,
                                           proposal_ids=prop_ids))

    def close(self):
        # type: () -> None
        """
        Close the session associated with this BundleDB.
        """
        self.session.close()
        self.session = None

    def is_open(self):
        # type: () -> bool
        """
        Return True iff the session associated with this BundleDB has not
        been closed.
        """
        return self.session is not None
