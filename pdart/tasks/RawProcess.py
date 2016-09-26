import multiprocessing


def exit_code_to_status(exit_code):
    """
    Convert the multiprocessing.Process.exitcode into a status string.
    """
    if exit_code is None:
        return 'RUNNING'
    elif exit_code == 0:
        return 'SUCCEEDED'
    elif exit_code > 0:
        return 'FAILED'
    elif exit_code < 0:
        return 'TERMINATED'


class RawProcess(object):
    """
    A wrapper around a Python multiprocessing.Process with a bit more
    instrumentation.
    """
    def __init__(self, target, args):
        assert target
        assert args is not None
        self.process = multiprocessing.Process(target=target, args=args)
        self.terminating = False

    def was_started(self):
        return self.process.pid is not None

    def start(self):
        assert not self.was_started()
        self.process.start()

    def is_alive(self):
        return self.process.is_alive()

    def status(self):
        if self.process.pid is None:
            return 'INITIALIZED'
        elif self.terminating and self.process.exitcode is None:
            return 'TERMINATING'
        status = exit_code_to_status(self.process.exitcode)
        if status == 'RUNNING':
            # check for timeout
            pass
        return status

    def join(self, timeout=None):
        assert self.was_started()
        self.process.join(timeout)

    def terminate(self):
        self.terminating = True
        self.process.terminate()
