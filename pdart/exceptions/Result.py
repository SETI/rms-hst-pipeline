"""
**You do not need to understand this module unless you want to
understand the implementations of** :mod:`pdart.exception.Combinators`
**or** :mod:`pdart.exception.ExceptionInfo` **.**

Normal Python functions either return a result value (possibly
``None``) or raise an exception which breaks the normal flow of the
code.  If we want to teach Python how to run multiple implementations
of a function or to remember multiple exceptions, we need to both
cases uniformly so we can use Python's data-handling capacity to
handle them.

We do this by wrapping function results: if it's a normal function
return, it's wrapped by :class:`~pdart.exception.Result.Success`; if
the function raised an exception, the exception and its stack trace
are wrapped in :class:`~pdart.exception.Result.Failure`.  Both are
subclasses of :class:`~pdart.exception.Result.Result`.

The end-user should never see these classes: they are only used
internally to implement functions like
:func:`~pdart.exception.Combinators.multiple_implementations`.
Results are wrapped to let Python work uniformly on all cases, then
the final result is unwrapped before presenting it to the user.
"""
import abc

import pdart.exceptions.ExceptionInfo


class Result(object):
    """
    The result of running a function: either
    :class:`~pdart.exception.Result.Success` or
    :class:`~pdart.exception.Result.Failure`.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_success(self):
        """
        Return True if the Result is a
        :class:`~pdart.exception.Result.Success`.
        """
        pass

    def is_failure(self):
        """
        Return True if the Result is a
        :class:`~pdart.exception.Result.Failure`.
        """
        return not self.is_success()


class Failure(Result):
    """
    The result of running a function and failing: a wrapper around
    :class:`~pdart.exception.ExceptionInfo.ExceptionInfo`.
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
