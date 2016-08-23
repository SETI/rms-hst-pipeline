import abc


class Result(object):
    """
    The result of running a function: either
    :class:`pdart.exception.Success` or
    :class:`pdart.exception.Failure`.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_success(self):
        """Return True if the Result is a Success"""
        pass

    def is_failure(self):
        """Return True if the Result is a Failure"""
        return not self.is_success()


class Failure(Result):
    """
    The result of running a function and failing: a wrapper around
    :class:`pdart.exception.ExceptionInfo`.
    """
    def __init__(self, exception_info):
        Result.__init__(self)
        self.exception_info = exception_info

    def is_success(self):
        return False

    def __str__(self):
        return '_Failure(%s)' % (self.exception_info, )


class Success(Result):
    """
    The result of running a function successfully: a wrapper around
    the returned value.
    """
    def __init__(self, value):
        Result.__init__(self)
        self.value = value

    def is_success(self):
        return True

    def __str__(self):
        return 'Success(%s)' % (self.value, )
