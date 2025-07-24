##########################################################################################
# queue_manager/task_queue_db.py
#
# This file is related to SQLite task queue database created by sqlalchemy. All the
# database related operations are included here.
##########################################################################################
import os

from queue_manager.config import (DB_PATH,
                                  DB_URI)
from sqlalchemy import (create_engine,
                        func,
                        Column,
                        Float,
                        Integer,
                        String)
from sqlalchemy.orm import (sessionmaker,
                            declarative_base)

# engine = create_engine(DB_URI, pool_size=20, max_overflow=0)
engine = create_engine(DB_URI)
Base = declarative_base()

class TaskQueue(Base):
    """
    A database representation of the task queue. Each row represents the task queue of
    a proposal id & visit, and it will have columns of the proposal id, visit, task num
    (current task), task priority, task executing order, status (current task status), and
    task command. Each row will have an unique combination of proposal id & visit columns.
    These will make sure tasks for these cases can be run in parallel:

    - tasks of different proposal ids
    - tasks for different visits of the same propsal id
    """

    __tablename__ = 'task_queue'

    id = Column(Integer, primary_key=True, nullable=False)
    proposal_id = Column(String, nullable=False)
    visit = Column(String, nullable=False)
    task = Column(String, nullable=False)
    priority = Column(Integer, nullable=False)
    order =  Column(Integer, nullable=False)
    status = Column(Integer, nullable=False)
    cmd = Column(String, nullable=False)

    def __repr__(self) -> str:
        return (
            f'TaskQueue(proposal_id={self.proposal_id!r}'
            f', visit={self.visit!r})'
            f', task={self.task!r})'
            f', priority={self.priority!r})'
            f', order={self.order!r})'
            f', status={self.status!r})'
            f', cmd={self.cmd!r})'
        )

def drop_task_queue_table():
    """
    Drop the task queue & subprocess list tables in the database.
    """
    TaskQueue.__table__.drop(engine)

def create_task_queue_table():
    """
    Create a database with the task queue & subprocess list tables.
    """
    Base.metadata.create_all(engine)

def init_task_queue_table():
    """
    Initialize the database by dropping the existing task queue table, and create a new
    & empty one.
    """
    drop_task_queue_table()
    create_task_queue_table()
    # Make sure all entries are clear if the database exists
    erase_all_task_queue()

def db_exists():
    """
    Check if the database exists before performing CRUD to it. Return a boolean flag.
    """
    return os.path.exists(DB_PATH)

def add_a_task(proposal_id, visit, task, priority, order, status, cmd):
    """
    Add an entry of the given proposal id & visit with its task num and task status to
    the task queue table. If the proposal id exists in the table, we update the entry.

    Input:
        proposal_id    a proposal id of the task queue.
        visit          a two character visit or ''.
        task           a number represents the current task.
        priority       a number reporeents task priority.
        order          a number reporeents task executing order.
        status         the status of the current task, 0 is wating and 1 is running.
        cmd            the command to run the task.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    # Add a task for a given proposal id & visit if the proposal id & visit combo doesn't
    # exist in the table
    entry = session.query(TaskQueue).filter(
                                         TaskQueue.proposal_id==proposal_id,
                                         TaskQueue.visit==visit,
                                         TaskQueue.task==task,
                                     ).first()
    if entry is None:
        new_entry = TaskQueue(proposal_id=proposal_id,
                              visit=visit,
                              task=task,
                              priority=priority,
                              order=order,
                              status=status,
                              cmd=cmd)
        session.add(new_entry)
    else:
        # If the current or a later task has been queued, we return False. This is a
        # flag to avoid spawning duplicated subprocess
        if entry.order >= order:
            return False
        entry.task = task
        entry.priority = priority
        entry.order = order
        entry.status = status
        entry.cmd = cmd
    session.commit()
    session.close()

def update_a_task_status(proposal_id, visit, task, status):
    """
    Update an entry of the given proposal id & visit with its task num and task status
    to the task queue table.

    Input:
        proposal_id    a proposal id of the task queue.
        visit          two character visit.
        task           a number represents the current task.
        status         the status of the current task, 0 is wating and 1 is running.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    row = session.query(TaskQueue).filter(
                                       TaskQueue.proposal_id==proposal_id,
                                       TaskQueue.visit==visit,
                                       TaskQueue.task==task
                                   ).first()
    if row is not None:
        row.status = status
        session.commit()
        session.close()

def remove_a_task(proposal_id, visit, task):
    """
    Remove a task queue entry of the given proposal id, visit, and task to the task queue
    table.

    Input:
        proposal_id    a proposal id of the task queue.
        visit          two character visit.
        task           a number represents the current task.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    session.query(TaskQueue).filter(
                                 TaskQueue.proposal_id==proposal_id,
                                 TaskQueue.visit==visit,
                                 TaskQueue.task==task
                             ).delete()
    session.commit()
    session.close()

def remove_all_tasks_for_a_prog_id_and_visit(proposal_id, visit):
    """
    Remove all task queue entries of the given proposal id & visit to the task queue
    table.

    Input:
        proposal_id    a proposal id of the task queue.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    session.query(TaskQueue).filter(
                                 TaskQueue.proposal_id==proposal_id,
                                 TaskQueue.visit==visit
                             ).delete()
    session.commit()
    session.close()

def remove_all_tasks_for_a_prog_id(proposal_id):
    """
    Remove all task queue entries of the given proposal id to the task queue table.

    Input:
        proposal_id    a proposal id of the task queue.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    session.query(TaskQueue).filter(TaskQueue.proposal_id==proposal_id).delete()
    session.commit()
    session.close()

def erase_all_task_queue():
    """
    Remove all entries in the task queue table.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    session.query(TaskQueue).delete()
    session.commit()
    session.close()

def get_next_task_to_be_run():
    """
    Get the next task to be run from database. Return the table row entry.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    # Get the tasks with the highest priority & waiting status
    subquery = (session.query(func.max(TaskQueue.priority)).filter(TaskQueue.status==0)
                                                           .scalar_subquery())
    # Get the task with the highest priority & task num, this will prioritize finishing
    # a pipeline process over running tasks at early pipeline stage or starting a new
    # pipeline process.
    query = (session.query(TaskQueue).filter(
                                          TaskQueue.priority==subquery,
                                          TaskQueue.status==0
                                      )
                                     .order_by(TaskQueue.task.desc())
                                     .first())
    session.close()
    return query

def get_total_number_of_tasks():
    """
    Get the total number of tasks stored in task queue table. Return the count of the
    entries.
    """
    Session = sessionmaker(engine)
    session = Session()
    # return session.query(TaskQueue).filter(TaskQueue.status==0).count()
    total_tasks =  session.query(TaskQueue).count()
    session.close()
    return total_tasks

def is_a_task_done(proposal_id, visit, task):
    """
    Check if a specific task for a given proposal id, visit, and task is done. (remove
    from the database)

    Input:
        proposal_id    a proposal id of the task queue.
        visit          two character visit.
        task           a number represents the current task.
    """
    Session = sessionmaker(engine)
    session = Session()

    entry = session.query(TaskQueue).filter(
                                        TaskQueue.proposal_id==proposal_id,
                                        TaskQueue.visit==visit,
                                        TaskQueue.task==task
                                    ).first()
    session.close()
    return True if not entry else False
