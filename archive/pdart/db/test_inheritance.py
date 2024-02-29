"""
My current implementation of the SqlAlchemy pattern to store an object
hierarchy isn't working as advertised: subclasses get put into the
database as their supertypes, but fields specific to the subclasses
don't go in.  Unfortunately, the documentation is woefully incomplete
and the implementation doesn't have internal checks that get
triggered.

This file is to do some exploratory testing to figure out what's going
on.  Examples are minimal, to try to debug the issue(s).
"""

import unittest
from typing import Any, Optional

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base: Any = declarative_base()


def create_tables(engine: Optional[Any]) -> None:
    Base.metadata.create_all(engine)


class Mammal(Base):
    __tablename__ = "mammals"

    id = Column(Integer, primary_key=True, nullable=False)
    type = Column(String(16), nullable=False)
    __mapper_args__ = {"polymorphic_identity": "mammal", "polymorphic_on": type}


# Ah!  I hadn't marked the mammal_id field as a primary_key.  With it,
# it issues INSERT statements on both mammals and cats tables.
# Without it, only on cats.  Problem solved.  Well, my problem solved.
# SqlAlchemy's documentation is still lacking and that remains a
# problem.


class Cat(Mammal):
    __tablename__ = "cats"

    mammal_id = Column(
        Integer, ForeignKey("mammals.id"), primary_key=True, nullable=False
    )
    birds_killed = Column(Integer, nullable=False)
    scratches_people = Column(Boolean, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "cat"}


class Dog(Mammal):
    __tablename__ = "dogs"

    mammal_id = Column(
        Integer, ForeignKey("mammals.id"), primary_key=True, nullable=False
    )
    housebroken = Column(Boolean, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "dog"}


class Test_Inheritance(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///", echo=True)  # in memory
        create_tables(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def testDBInsertion(self) -> None:
        cat = Cat(birds_killed=23, scratches_people=True)
        self.session.add(cat)
        self.session.commit()
        cats = self.session.query(Cat).all()
        assert cats
