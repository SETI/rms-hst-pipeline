from sqlalchemy import Column, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from sqlalchemy.engine import Engine

Base = declarative_base()  # type: Any


def create_tables(engine):
    # type: (Engine) -> None
    Base.metadata.create_all(engine)


class UpdateDatetime(Base):
    __tablename__ = 'update_datetime'

    bundle_id = Column(String, primary_key=True, nullable=False)
    update_datetime = Column(DateTime, nullable=False)

    def __repr__(self):
        return 'UpdateDatetime(%r)' % self.update_datetime

    def __str__(self):
        return 'UpdateDatetime(%s)' % self.update_datetime


class CheckDatetime(Base):
    __tablename__ = 'check_datetime'

    check_datetime = Column(DateTime, primary_key=True, nullable=False)
    bundle_ids = Column(String, nullable=False)

    def __repr__(self):
        return 'CheckDatetime(%r, %r)' % (self.check_datetime,
                                          self.bundle_ids)

    def __str__(self):
        return 'CheckDatetime(%s, %s)' % (self.check_datetime,
                                          self.bundle_ids)
