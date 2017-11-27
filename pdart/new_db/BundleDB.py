from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pdart.new_db.SqlAlchTables import *

_BUNDLE_DB_NAME = 'bundle$database.db'


# type: unicode

class BundleDB(object):
    def __init__(self, url):
        # type: (unicode) -> None
        self.url = url
        self.engine = create_engine(url)
        self.session = sessionmaker(bind=self.engine)()

    @staticmethod
    def create_database_from_os_filepath(os_filepath):
        # type: (unicode) -> BundleDB
        return BundleDB('sqlite:///' + os_filepath)

    @staticmethod
    def create_database_in_memory():
        # type: () -> BundleDB
        return BundleDB('sqlite:///')

    def create_tables(self):
        # type: () -> None
        create_tables(self.engine)

    def close(self):
        # type: () -> None
        self.session.close()
        self.session = None

    def is_open(self):
        # type: () -> bool
        return self.session is not None
