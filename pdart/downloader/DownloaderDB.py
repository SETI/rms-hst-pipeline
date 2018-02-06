import datetime

from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker

from pdart.downloader.SqlAlchTables import *

if TYPE_CHECKING:
    from typing import Set, Tuple


def _string_to_str_set(str):
    # type: (str) -> Set[str]
    return set(str.split())


def _str_set_to_string(strset):
    # type: (Set[str]) -> str
    return ' '.join(sorted(list(strset)))


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

    def get_last_update_datetime(self, bundle):
        # type: (str) -> datetime.datetime
        return self.session.query(
            UpdateDatetime).filter(
            UpdateDatetime.bundle_id == bundle).one().update_datetime

    def set_last_update_datetime(self, bundle, last):
        # type: (str, datetime.datetime) -> None
        if self.session.query(
                exists().where(
                    UpdateDatetime.bundle_id == bundle)).scalar():
            # No need for locking the row here since I'm not modifying
            # the value, just replacing it.
            self.session.query(
                UpdateDatetime).update({'bundle_id': bundle,
                                        'update_datetime': last})
        else:
            self.session.add(UpdateDatetime(bundle_id=bundle,
                                            update_datetime=last))

    def get_last_check(self):
        # type: () -> Tuple[datetime.datetime, Set[str]]
        check = self.session.query(
            CheckDatetime).one()
        return (check.check_datetime, _string_to_str_set(check.bundle_ids))

    def set_last_check(self, last, bundle_set):
        # type: (datetime.datetime, Set[str]) -> None
        bundles = _str_set_to_string(bundle_set)
        if self.session.query(CheckDatetime).count():
            # No need for locking the row here since I'm not modifying
            # the value, just replacing it.
            self.session.query(
                CheckDatetime).update({'check_datetime': last,
                                       'bundle_ids': bundles})
        else:
            self.session.add(CheckDatetime(check_datetime=last,
                                           bundle_ids=bundles))

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
