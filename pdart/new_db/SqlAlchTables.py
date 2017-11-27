from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from sqlalchemy.engine import Engine

Base = declarative_base()  # type: Any


class Bundle(Base):
    __tablename__ = 'bundles'
    lidvid = Column(String, primary_key=True, nullable=False)


def create_tables(engine):
    # type: (Engine) -> None
    Base.metadata.create_all(engine)
