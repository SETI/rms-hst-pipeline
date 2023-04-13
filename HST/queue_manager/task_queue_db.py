##########################################################################################
# queue_manager/task_queue_db.py
##########################################################################################
from sqlalchemy import (create_engine,
                        func,
                        MetaData,
                        Table,
                        Column,
                        Integer,
                        String)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from hst_helper import DB_URI

engine = create_engine(DB_URI, echo = True)
Base = declarative_base()

class TaskQueue(Base):
    """
    A database representation of the task queue. Each row represents the task queue of
    a proposal id, and it will have columns of the proposal id, task num (current task),
    task priority and status (current task status). Each row will have unique proposal
    id column.
    """

    __tablename__ = "tq"

    id = Column(Integer, primary_key=True, nullable=False)
    proposal_id = Column(String, nullable=False)
    task_num = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False)
    status = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return (
            f"TaskQueue(proposal_id={self.proposal_id!r}"
            f", task_num={self.task_num!r})"
            f", priority={self.priority!r})"
            f", status={self.status!r})"
        )

def drop_task_queue_table():
    TaskQueue.__table__.drop(engine)

def create_task_queue_table():
    Base.metadata.create_all(engine)

def init_task_queue_table():
    drop_task_queue_table()
    create_task_queue_table()
    # Make sure all entries are clear if the database exists
    erase_all_task_queue()

def add_a_prog_id_task_queue(proposal_id, task_num, priority, status):
    """
    Add an entry of the given proposal id with its task num and task status to the task
    queue table. If the proposal id exists in the table, we update the entry.
    Input:
        proposal_id:   a proposal id of the task queue.
        task_num:      a number represents the current task.
        priority:      a number reporeents task priority.
        status:        the status of the current task, 0 is wating and 1 is running.
    """

    Session = sessionmaker(engine)
    session = Session()
    # Add a task for a given proposal id if the proposal id doesn't exist in the table
    entry = session.query(TaskQueue).filter(TaskQueue.proposal_id==proposal_id).first()
    if entry is None:
        new_entry = TaskQueue(proposal_id=proposal_id,
                          task_num=task_num,
                          priority=priority,
                          status=status)
        session.add(new_entry)
    else:
        # If the current or later task has been queued, we return False. This is a flag to
        # avoid spawning duplicated subprocess
        if entry.task_num >= task_num:
            return False
        entry.task_num = task_num
        entry.priority = priority
        entry.status = status
    session.commit()

def update_a_prog_id_task_queue(proposal_id, task_num, priority, status):
    """
    Update an entry of the given proposal id with its task num and task status to the
    task queue table.
    Input:
        proposal_id:   a proposal id of the task queue.
        task_num:      a number represents the current task.
        priority:      a number reporeents task priority.
        status:        the status of the current task, 0 is wating and 1 is running.
    """
    Session = sessionmaker(engine)
    session = Session()
    row = session.query(TaskQueue).filter(TaskQueue.proposal_id==proposal_id).first()
    if row is not None:
        row.task_num = task_num
        row.priority = priority
        row.status = status
        session.commit()

def update_a_prog_id_task_status(proposal_id, status):
    """
    Update an entry of the given proposal id with its task num and task status to the
    task queue table.
    Input:
        proposal_id:   a proposal id of the task queue.
        status:        the status of the current task, 0 is wating and 1 is running.
    """
    Session = sessionmaker(engine)
    session = Session()
    row = session.query(TaskQueue).filter(TaskQueue.proposal_id==proposal_id).first()
    if row is not None:
        row.status = status
        session.commit()

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

def get_next_task_to_be_run():
    """
    Get the next task to be run from database. Return the table row entry.
    """
    Session = sessionmaker(engine)
    session = Session()
    # Get the tasks with the highest priority & waiting status
    subquery = session.query(func.max(TaskQueue.priority)).filter(TaskQueue.status==0)
    # Get the task with the highest priority & task num, this will prioritize finishing
    # a pipeline process over running tasks at early pipeline stage or starting a new
    # pipeline process.
    query = (session.query(TaskQueue).filter(
                                        TaskQueue.priority==subquery,
                                        TaskQueue.status==0
                                      )
                                     .order_by(TaskQueue.task_num.desc())
                                     .first())
    return query
