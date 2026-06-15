##########################################################################################
# queue_manager/task_queue_db.py
#
# This file is related to SQLite task queue database created by sqlalchemy. All the
# database related operations are included here.
##########################################################################################
import datetime
import os

from queue_manager.config import (DB_PATH,
                                  DB_URI,
                                  LOWER_LVL_TASKS)
from sqlalchemy import (create_engine,
                        Column,
                        DateTime,
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
    (current task), task priority, task executing order, status (current task status),
    task command, and execution time. Each row will have an unique combination of proposal
    id & visit columns.
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
    execution_time = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f'TaskQueue(proposal_id={self.proposal_id!r}'
            f', visit={self.visit!r})'
            f', task={self.task!r})'
            f', priority={self.priority!r})'
            f', order={self.order!r})'
            f', status={self.status!r})'
            f', cmd={self.cmd!r})'
            f', execution_time={self.execution_time!r})'
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

def _can_ignore_later_finalize_bundle(later_entries):
    """
    Return True if the only higher-order queued task is a waiting finalize_bundle
    with a scheduled execution_time.
    """
    return (len(later_entries) == 1
            and later_entries[0].task == 'finalize_bundle'
            and later_entries[0].execution_time is not None
            and later_entries[0].status == 0)

def add_a_task(proposal_id, visit, task, priority, order, status, cmd, execution_time=None):
    """
    Add an entry for the given proposal id, visit, and task to the task queue table.

    Returns False if the task is already queued, or if another task for the same
    proposal id and visit has a higher order (a later pipeline stage is already queued),
    unless the only such task is a waiting finalize_bundle with a scheduled
    execution_time. Otherwise adds the entry and returns None.

    If task is finalize_bundle and one already exists for the proposal id, the existing
    row is updated in place (including execution_time) instead of returning False.

    Input:
        proposal_id      a proposal id of the task queue.
        visit            a two character visit or ''.
        task             a string represents the current task.
        priority         a number reporeents task priority.
        order            a number reporeents task executing order.
        status           the status of the current task, 0 is wating and 1 is running.
        cmd              the command to run the task.
        execution_time   the date/time when the task becomes eligible to run, or None.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    if task == 'finalize_bundle':
        entry = session.query(TaskQueue).filter(
            TaskQueue.proposal_id == proposal_id,
            TaskQueue.task == 'finalize_bundle',
        ).first()
        if entry is not None:
            entry.visit = visit
            entry.priority = priority
            entry.order = order
            entry.status = status
            entry.cmd = cmd
            entry.execution_time = execution_time
            session.commit()
            session.close()
            return None

    # Check if the task is queued
    entry = session.query(TaskQueue).filter(
        TaskQueue.proposal_id == proposal_id,
        TaskQueue.visit == visit,
        TaskQueue.task == task,
    ).first()
    if entry is not None:
        session.close()
        return False

    # Check if a task with higher order is queued
    later_entries = session.query(TaskQueue).filter(
        TaskQueue.proposal_id == proposal_id,
        TaskQueue.visit == visit,
        TaskQueue.order > order,
    ).all()
    if later_entries and not _can_ignore_later_finalize_bundle(later_entries):
        session.close()
        return False

    session.add(TaskQueue(proposal_id=proposal_id,
                          visit=visit,
                          task=task,
                          priority=priority,
                          order=order,
                          status=status,
                          cmd=cmd,
                          execution_time=execution_time))
    session.commit()
    session.close()
    return None

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

def queue_cleanup_during_restart():
    """
    Reset the task queue after a restart: drop rows for tasks that must be re-queued from
    their program/visit entry points, and clear any 'running' (status 1) flags so every
    remaining row is waiting (status 0). Does nothing if the database file is missing or
    the queue table has no rows.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    with Session.begin() as session:
        if session.query(TaskQueue).count() == 0:
            return

        session.query(TaskQueue).filter(TaskQueue.task.in_(LOWER_LVL_TASKS)).delete(
            synchronize_session=False
        )
        session.query(TaskQueue).filter(TaskQueue.status == 1).update(
            {TaskQueue.status: 0},
            synchronize_session=False,
        )

def _task_is_ready_to_run(task, now):
    """
    Return True if the task has no execution_time or the current time has reached it.
    """
    return task.execution_time is None or now >= task.execution_time

def get_next_task_to_be_run():
    """
    Get the next task to be run from database. Return the table row entry.

    Waiting tasks are considered in descending priority and order. Tasks with a future
    execution_time are skipped until that time has passed.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    now = datetime.datetime.now()
    waiting_tasks = (session.query(TaskQueue).filter(TaskQueue.status == 0)
                     .order_by(TaskQueue.priority.desc(), TaskQueue.order.desc())
                     .all())
    for task in waiting_tasks:
        if _task_is_ready_to_run(task, now):
            session.close()
            return task
    session.close()
    return None

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
