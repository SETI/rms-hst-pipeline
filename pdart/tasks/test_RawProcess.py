from pdart.tasks.RawProcess import *

import os
import signal
import time


def test_exit_code_to_status():
    assert exit_code_to_status(None) == 'RUNNING'
    assert exit_code_to_status(0) == 'SUCCEEDED'
    assert exit_code_to_status(-1) == 'TERMINATED'
    assert exit_code_to_status(1) == 'FAILED'


def _pass():
    pass


def test_RawProcess_returning():
    # Test one: process immediately returns
    rp = RawProcess(target=_pass, args=tuple())
    assert rp
    assert not rp.was_started()
    assert not rp.is_alive()
    assert rp.status() == 'INITIALIZED'

    rp.start()
    assert rp.was_started()
    # might still be running; might not
    stat = rp.status()
    assert stat in ['RUNNING', 'SUCCEEDED']
    if stat == 'SUCCEEDED':
        assert not rp.is_alive()

    rp.join()
    # is definitely not running
    stat = rp.status()
    assert stat == 'SUCCEEDED'
    assert not rp.is_alive()


def _sleep(secs):
    time.sleep(secs)


def test_RawProcess_terminate():
    # Test two: process hangs
    rp = RawProcess(target=_sleep, args=(60,))
    rp.start()
    assert rp.status() == 'RUNNING'
    rp.terminate()
    rp.join()
    assert rp.status() == 'TERMINATED'


def _munge():
    while True:
        time.sleep(10)


def test_RawProcess_terminating():
    # Test three: process takes a while to terminate.  Check that it
    # shows 'TERMINATING' status before it goes to 'TERMINATED'.

    JOIN_SECS = 10 * 1.0 / 1000  # 10 ms

    # The signal handler needs to be set *before* we fork the
    # process
    original_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    rp = RawProcess(target=_munge, args=tuple())
    rp.start()
    signal.signal(signal.SIGTERM, original_handler)

    # Try to end the process (but it'll ignore the request and keep on
    # going)
    rp.terminate()
    assert rp.status() == 'TERMINATING'
    rp.join(JOIN_SECS)
    # it still hasn't terminated even after we wait
    assert rp.status() == 'TERMINATING'

    # Send a different signal to terminate it with extreme prejudice
    os.kill(rp.process.pid, signal.SIGKILL)

    rp.join(JOIN_SECS)
    # Now it's dead
    assert rp.status() == 'TERMINATED'
