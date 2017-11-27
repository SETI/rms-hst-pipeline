from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pdart.new_db.SqlAlchTables import *

_BUNDLE_DB_NAME = 'bundle$database.db'


# type: unicode

class BundleDB(object):
    def __init__(self, os_filepath):
        # type: (unicode) -> None
        self.os_filepath = os_filepath
        url = 'sqlite:///' + os_filepath
        self.engine = create_engine(url)
        self.session = sessionmaker(bind=self.engine)()

    def create_tables(self):
        create_tables(self.engine)

    def close(self):
        # type: () -> None
        self.session.close()
        self.session = None

    def is_open(self):
        # type: () -> bool
        return self.session is not None
