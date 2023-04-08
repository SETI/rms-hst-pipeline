##########################################################################################
# queue_manager/task_queue_db.py
##########################################################################################
from sqlalchemy import (create_engine,
                        MetaData,
                        Table,
                        Column,
                        Integer,
                        String)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from hst_helper import DB_URI

# def create_task_queue_table():
#     engine = create_engine(DB_URI)
#     # Create a metadata instance
#     metadata = MetaData(engine)
#     # Declare a table
#     table = Table('tq',metadata,
#                   Column('id',Integer, primary_key=True, nullable=False),
#                   Column('proposal_id',String, nullable=False),
#                   Column('task_num',Integer, nullable=False),
#                   Column('status',Integer, nullable=False))
#     # Create task queue table
#     table.create()
engine = create_engine(DB_URI, echo = True)
Base = declarative_base()

class TaskQueue(Base):
    """
    A database representation of the task queue. Each row represents the task queue of
    a proposal id, and it will have columns of the proposal id, task num (current task),
    and status (current task status). Each row will have unique proposal id column.
    """

    __tablename__ = "tq"

    id = Column(Integer, primary_key=True, nullable=False)
    proposal_id = Column(String, nullable=False)
    task_num = Column(Integer, nullable=False)
    status = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return (
            f"TaskQueue(proposal_id={self.proposal_id!r}"
            f", task_num={self.task_num!r})"
            f", status={self.status!r})"
        )

def create_task_queue_table():
    Base.metadata.create_all(engine)

def add_a_prog_id_task_queue(proposal_id, task_num, status):
    """
    Add an entry of the given proposal id with its task num and task status to the task
    queue table. If the proposal id exists in the table, we don't add the entry.
    Input:
        proposal_id:   a proposal id of the task queue.
        task_num:      a number represents the current task.
        status:        the status of the current task, 0 is wating and 1 is running.
    """
    new_entry = TaskQueue(proposal_id=proposal_id,
                          task_num=task_num,
                          status=status)
    Session = sessionmaker(engine)
    session = Session()
    # Add a task for a given proposal id if the proposal id doesn't exist in the table
    if (session.query(TaskQueue.id)
        .filter(TaskQueue.proposal_id==proposal_id)
        .first() is None):
        session.add(new_entry)
        session.commit()

def update_a_prog_id_task_queue(proposal_id, task_num, status):
    """
    Update an entry of the given proposal id with its task num and task status to the
    task queue table.
    Input:
        proposal_id:   a proposal id of the task queue.
        task_num:      a number represents the current task.
        status:        the status of the current task, 0 is wating and 1 is running.
    """
    Session = sessionmaker(engine)
    session = Session()
    row = session.query(TaskQueue).filter(TaskQueue.proposal_id==proposal_id).first()
    # print('original:', row.task_num, row.status)
    # Update the task queue entry if it exists the table
    if row is not None:
        row.task_num = task_num
        row.status = status
        session.commit()

    # row = session.query(TaskQueue).filter(TaskQueue.proposal_id==proposal_id).first()
    # print('updated:', row.task_num, row.status)

def remove_a_prog_id_task_queue(proposal_id):
    """
    Remove a task queue entry of the given proposal id to the task queue table.
    Input:
        proposal_id:   a proposal id of the task queue.
    """
    Session = sessionmaker(engine)
    session = Session()
    session.query(TaskQueue).filter(TaskQueue.proposal_id==proposal_id).delete()
    session.commit()

def erase_all_task_queue():
    """
    Remove all entries in the task queue table.
    """
    Session = sessionmaker(engine)
    session = Session()
    session.query(TaskQueue).delete()
    session.commit()
