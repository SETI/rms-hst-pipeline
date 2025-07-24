import os
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from queue_manager import task_queue_db

def setup_module(module):
    # Use a temporary SQLite DB for testing
    module._old_db_path = task_queue_db.DB_PATH
    module._old_db_uri = task_queue_db.DB_URI
    module.temp_db_fd, module.temp_db_path = tempfile.mkstemp(suffix='.sqlite')
    task_queue_db.DB_PATH = module.temp_db_path
    task_queue_db.DB_URI = f'sqlite:///{module.temp_db_path}'
    task_queue_db.engine = create_engine(task_queue_db.DB_URI)
    task_queue_db.Base.metadata.bind = task_queue_db.engine

def teardown_module(module):
    os.close(module.temp_db_fd)
    os.remove(module.temp_db_path)
    task_queue_db.DB_PATH = module._old_db_path
    task_queue_db.DB_URI = module._old_db_uri
    task_queue_db.engine = create_engine(task_queue_db.DB_URI)
    task_queue_db.Base.metadata.bind = task_queue_db.engine

def test_table_creation_and_existence():
    task_queue_db.create_task_queue_table()
    assert task_queue_db.db_exists()
    # Table should exist
    assert task_queue_db.engine.dialect.has_table(task_queue_db.engine.connect(), 'task_queue')

def test_add_and_query_task():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    Session = sessionmaker(task_queue_db.engine)
    session = Session()
    entry = session.query(task_queue_db.TaskQueue).filter_by(proposal_id='123', visit='A1', task='T1').first()
    assert entry is not None
    assert entry.priority == 5
    assert entry.status == 0
    session.close()

def test_update_a_task_status():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    task_queue_db.update_a_task_status('123', 'A1', 'T1', 1)
    Session = sessionmaker(task_queue_db.engine)
    session = Session()
    entry = session.query(task_queue_db.TaskQueue).filter_by(proposal_id='123', visit='A1', task='T1').first()
    assert entry.status == 1
    session.close()

def test_remove_a_task():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    task_queue_db.remove_a_task('123', 'A1', 'T1')
    Session = sessionmaker(task_queue_db.engine)
    session = Session()
    entry = session.query(task_queue_db.TaskQueue).filter_by(proposal_id='123', visit='A1', task='T1').first()
    assert entry is None
    session.close()

def test_remove_all_tasks_for_a_prog_id_and_visit():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    task_queue_db.add_a_task('123', 'A1', 'T2', 5, 2, 0, 'echo test2')
    task_queue_db.remove_all_tasks_for_a_prog_id_and_visit('123', 'A1')
    Session = sessionmaker(task_queue_db.engine)
    session = Session()
    entries = session.query(task_queue_db.TaskQueue).filter_by(proposal_id='123', visit='A1').all()
    assert len(entries) == 0
    session.close()

def test_remove_all_tasks_for_a_prog_id():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    task_queue_db.add_a_task('123', 'B2', 'T2', 5, 2, 0, 'echo test2')
    task_queue_db.remove_all_tasks_for_a_prog_id('123')
    Session = sessionmaker(task_queue_db.engine)
    session = Session()
    entries = session.query(task_queue_db.TaskQueue).filter_by(proposal_id='123').all()
    assert len(entries) == 0
    session.close()

def test_erase_all_task_queue():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    task_queue_db.add_a_task('456', 'B2', 'T2', 5, 2, 0, 'echo test2')
    task_queue_db.erase_all_task_queue()
    Session = sessionmaker(task_queue_db.engine)
    session = Session()
    entries = session.query(task_queue_db.TaskQueue).all()
    assert len(entries) == 0
    session.close()

def test_get_next_task_to_be_run():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    task_queue_db.add_a_task('456', 'B2', 'T2', 10, 2, 0, 'echo test2')
    next_task = task_queue_db.get_next_task_to_be_run()
    assert next_task is not None
    assert next_task.proposal_id == '456'
    assert next_task.priority == 10

def test_get_total_number_of_tasks():
    task_queue_db.init_task_queue_table()
    assert task_queue_db.get_total_number_of_tasks() == 0
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    assert task_queue_db.get_total_number_of_tasks() == 1
    task_queue_db.add_a_task('456', 'B2', 'T2', 10, 2, 0, 'echo test2')
    assert task_queue_db.get_total_number_of_tasks() == 2

def test_is_a_task_done():
    task_queue_db.init_task_queue_table()
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    assert not task_queue_db.is_a_task_done('123', 'A1', 'T1')
    task_queue_db.remove_a_task('123', 'A1', 'T1')
    assert task_queue_db.is_a_task_done('123', 'A1', 'T1')

def test_add_duplicate_task_order():
    task_queue_db.init_task_queue_table()
    # Add a task
    task_queue_db.add_a_task('123', 'A1', 'T1', 5, 1, 0, 'echo test')
    # Try to add the same task with a lower order (should not update)
    result = task_queue_db.add_a_task('123', 'A1', 'T1', 5, 0, 0, 'echo test')
    assert result is False
    # Try to add the same task with a higher order (should update)
    result2 = task_queue_db.add_a_task('123', 'A1', 'T1', 6, 2, 1, 'echo test2')
    assert result2 is None
    Session = sessionmaker(task_queue_db.engine)
    session = Session()
    entry = session.query(task_queue_db.TaskQueue).filter_by(proposal_id='123', visit='A1', task='T1').first()
    assert entry.priority == 6
    assert entry.order == 2
    assert entry.status == 1
    assert entry.cmd == 'echo test2'
    session.close()

def test_repr():
    t = task_queue_db.TaskQueue(proposal_id='p', visit='v', task='t', priority=1, order=1, status=0, cmd='c')
    assert 'TaskQueue' in repr(t)

def test_drop_task_queue_table():
    task_queue_db.create_task_queue_table()
    # Table should exist
    assert task_queue_db.engine.dialect.has_table(task_queue_db.engine.connect(), 'task_queue')
    task_queue_db.drop_task_queue_table()
    # Table should not exist
    assert not task_queue_db.engine.dialect.has_table(task_queue_db.engine.connect(), 'task_queue')

def test_db_exists_true_false(tmp_path):
    # Should be true for the temp DB
    task_queue_db.create_task_queue_table()
    assert task_queue_db.db_exists() is True
    # Should be false for a non-existent file
    old_path = task_queue_db.DB_PATH
    task_queue_db.DB_PATH = str(tmp_path / 'nonexistent.sqlite')
    assert task_queue_db.db_exists() is False
    task_queue_db.DB_PATH = old_path

def test_add_a_task_db_missing(monkeypatch):
    monkeypatch.setattr(task_queue_db, 'db_exists', lambda: False)
    result = task_queue_db.add_a_task('p', 'v', 't', 1, 1, 0, 'c')
    assert result is None

def test_update_a_task_status_db_missing(monkeypatch):
    monkeypatch.setattr(task_queue_db, 'db_exists', lambda: False)
    assert task_queue_db.update_a_task_status('p', 'v', 't', 1) is None

def test_remove_a_task_db_missing(monkeypatch):
    monkeypatch.setattr(task_queue_db, 'db_exists', lambda: False)
    assert task_queue_db.remove_a_task('p', 'v', 't') is None

def test_remove_all_tasks_for_a_prog_id_and_visit_db_missing(monkeypatch):
    monkeypatch.setattr(task_queue_db, 'db_exists', lambda: False)
    assert task_queue_db.remove_all_tasks_for_a_prog_id_and_visit('p', 'v') is None

def test_remove_all_tasks_for_a_prog_id_db_missing(monkeypatch):
    monkeypatch.setattr(task_queue_db, 'db_exists', lambda: False)
    assert task_queue_db.remove_all_tasks_for_a_prog_id('p') is None

def test_erase_all_task_queue_db_missing(monkeypatch):
    monkeypatch.setattr(task_queue_db, 'db_exists', lambda: False)
    assert task_queue_db.erase_all_task_queue() is None

def test_get_next_task_to_be_run_db_missing(monkeypatch):
    monkeypatch.setattr(task_queue_db, 'db_exists', lambda: False)
    assert task_queue_db.get_next_task_to_be_run() is None
