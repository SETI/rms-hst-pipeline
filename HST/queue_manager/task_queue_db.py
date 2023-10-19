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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine(DB_URI, echo=True)
Base = declarative_base()

class TaskQueue(Base):
    """
    A database representation of the task queue. Each row represents the task queue of
    a proposal id & visit, and it will have columns of the proposal id, visit, task num
    (current task), task priority, status (current task status), and task command. Each
    row will have an unique combination of proposal id & visit columns. These will make
    sure tasks for these cases can be run in parallel:

    - tasks of different proposal ids
    - tasks for different visits of the same propsal id
    """

    __tablename__ = 'task_queue'

    id = Column(Integer, primary_key=True, nullable=False)
    proposal_id = Column(String, nullable=False)
    visit = Column(String, nullable=False)
    task_num = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False)
    status = Column(Integer, nullable=False)
    cmd = Column(String, nullable=False)

    def __repr__(self) -> str:
        return (
            f'TaskQueue(proposal_id={self.proposal_id!r}'
            f', visit={self.visit!r})'
            f', task_num={self.task_num!r})'
            f', priority={self.priority!r})'
            f', status={self.status!r})'
            f', cmd={self.cmd!r})'
        )

class SubprocessList(Base):
    """
    A database representation of the subprocess list. Each row represents the subprocess
    being run by subprocess.Popen(), and it will have columns of pid, task_num, start
    time, max end time (start time + max_allowed_time), visit and proposal_id.
    """

    __tablename__ = 'subprocess_list'

    id = Column(Integer, primary_key=True, nullable=False)
    pid = Column(Integer, nullable=False)
    task_num = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    max_end_time = Column(Float, nullable=False)
    visit = Column(String, nullable=False)
    proposal_id = Column(String, nullable=False)

    def __repr__(self) -> str:
        return (
            f'SubprocessList(pid={self.pid!r}'
            f', task_num={self.task_num!r})'
            f', start_time={self.start_time!r})'
            f', max_end_time={self.max_end_time!r})'
            f', visit={self.visit!r})'
            f', proposal_id={self.proposal_id!r})'
        )

def drop_task_queue_table():
    """
    Drop the task queue & subprocess list tables in the database.
    """
    TaskQueue.__table__.drop(engine)
    SubprocessList.__table__.drop(engine)

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

def add_a_prog_id_task_queue(proposal_id, visit, task_num, priority, status, cmd):
    """
    Add an entry of the given proposal id & visit with its task num and task status to
    the task queue table. If the proposal id exists in the table, we update the entry.

    Input:
        proposal_id    a proposal id of the task queue.
        visit          a two character visit or ''.
        task_num       a number represents the current task.
        priority       a number reporeents task priority.
        status         the status of the current task, 0 is wating and 1 is running.
        cmd            the command to run the task.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    # Add a task for a given proposal id if the proposal id doesn't exist in the table
    entry = session.query(TaskQueue).filter(
                                         TaskQueue.proposal_id==proposal_id,
                                         TaskQueue.visit==visit
                                     ).first()
    if entry is None:
        new_entry = TaskQueue(proposal_id=proposal_id,
                              visit=visit,
                              task_num=task_num,
                              priority=priority,
                              status=status,
                              cmd=cmd)
        session.add(new_entry)
    else:
        # If the current or a later task has been queued, we return False. This is a
        # flag to avoid spawning duplicated subprocess
        if entry.task_num >= task_num:
            return False
        entry.task_num = task_num
        entry.priority = priority
        entry.status = status
        entry.cmd = cmd
    session.commit()

def update_a_prog_id_task_status(proposal_id, visit, status):
    """
    Update an entry of the given proposal id & visit with its task num and task status
    to the task queue table.

    Input:
        proposal_id    a proposal id of the task queue.
        visit          two character visit.
        status         the status of the current task, 0 is wating and 1 is running.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    row = session.query(TaskQueue).filter(
                                       TaskQueue.proposal_id==proposal_id,
                                       TaskQueue.visit==visit
                                   ).first()
    if row is not None:
        row.status = status
        session.commit()

def remove_a_prog_id_task_queue(proposal_id, visit):
    """
    Remove a task queue entry of the given proposal id & visit to the task queue table.
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

def remove_all_task_queue_for_a_prog_id(proposal_id):
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

def erase_all_task_queue():
    """
    Remove all entries in the task queue table.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    session.query(TaskQueue).delete()
    session.query(SubprocessList).delete()
    session.commit()

def get_next_task_to_be_run():
    """
    Get the next task to be run from database. Return the table row entry.
    """
    if not db_exists():
        return

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

def add_a_subprocess(pid, task_num, start_time, max_end_time, visit, proposal_id):
    """
    Add an entry of the subprocess to the subprocess list table. If the pid exists in the
    table, we just exit.

    Input:
        pid             the subprocess.Popen instance created when a subprocess is run.
        task_num        a number represents the current task.
        start_time      the start time of a subprocess.
        max_end_time    the max possible end time of a subprocess.
        visit           two character visit or ''.
        proposal_id     a proposal id.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    # Add an entry of the subprocess to the subprocess list table if the subprocess task,
    # visit, and proposal id don't exist.
    entry = session.query(SubprocessList).filter(
                                              SubprocessList.task_num==task_num,
                                              SubprocessList.visit==visit,
                                              SubprocessList.proposal_id==proposal_id
                                          ).first()
    if entry is None:
        new_entry = SubprocessList(pid=pid,
                                   task_num=task_num,
                                   start_time=start_time,
                                   max_end_time=max_end_time,
                                   visit=visit,
                                   proposal_id=proposal_id)
        session.add(new_entry)
        session.commit()

def get_pid_by_prog_id_task_and_visit(proposal_id, task_num, visit):
    """
    Get the subprocess id with the given task_num ,visit & proposal_id

    Input:
        proposal_id     a proposal id.
        task_num        a number represents the current task.
        visit           two character visit or ''.
    """
    Session = sessionmaker(engine)
    session = Session()
    if visit != '':
        query = session.query(SubprocessList).filter(
                                                  SubprocessList.task_num==task_num,
                                                  SubprocessList.visit==visit,
                                                  SubprocessList.proposal_id==proposal_id
                                              )
    else:
        query = session.query(SubprocessList).filter(
                                                  SubprocessList.task_num==task_num,
                                                  SubprocessList.proposal_id==proposal_id
                                              )
    return [subproc.pid for subproc in query]

def get_total_number_of_subprocesses():
    """
    Get the total number of subprocesses stored in subprocess list table. Return the
    count of the rows.
    """
    Session = sessionmaker(engine)
    session = Session()
    return session.query(SubprocessList).count()


def get_all_subprocess_info():
    """
    Get a list of all the subprocess info. Return the list of subprocess info.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    all_subproc = session.query(SubprocessList)
    return ([(subproc.pid, subproc.task_num, subproc.start_time,
              subproc.max_end_time, subproc.visit, subproc.proposal_id)
              for subproc in all_subproc])

def remove_a_subprocess_by_pid(pid):
    """
    Remove a subprocess entry of the process id in the subprocess list table.

    Input:
        pid:    a process id
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    session.query(SubprocessList).filter(SubprocessList.pid==pid).delete()
    session.commit()

def remove_a_subprocess_by_prog_id_task_and_visit(proposal_id, task_num, visit):
    """
    Remove a subprocess entry of the given task, visit, and proposal id in the subprocess
    list table.

    Input:
        proposal_id     a proposal id.
        task_num        a number represents the current task.
        visit           two character visit or ''.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    if visit != '':
        session.query(SubprocessList).filter(SubprocessList.task_num==task_num,
                                             SubprocessList.visit==visit,
                                             SubprocessList.proposal_id==proposal_id
                                            ).delete()
    else:
        session.query(SubprocessList).filter(SubprocessList.task_num==task_num,
                                             SubprocessList.proposal_id==proposal_id
                                            ).delete()
    session.commit()

def remove_all_subprocess_for_a_prog_id(proposal_id):
    """
    Remove all subprocess entries of the given proposal id to the subprocess list table.

    Input:
        proposal_id    a proposal id of the task queue.
    """
    if not db_exists():
        return

    Session = sessionmaker(engine)
    session = Session()
    session.query(SubprocessList).filter(SubprocessList.proposal_id==proposal_id).delete()
    session.commit()
