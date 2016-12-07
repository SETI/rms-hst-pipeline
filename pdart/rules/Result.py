"""
**New to PDART?** You do not need to understand this module unless you
want to understand the implementations of
:mod:`pdart.rules.Combinators` or :mod:`pdart.rules.ExceptionInfo`.

Normal Python functions either return a result value (possibly
``None``) or raise an exception which breaks the normal flow of the
code.  If we want to teach Python how to run multiple implementations
of a function or to remember multiple exceptions, we need to both
cases uniformly so we can use Python's data-handling capacity to
handle them.

We do this by wrapping function results: if it's a normal function
return, it's wrapped by :class:`~pdart.rules.Result.Success`; if
the function raised an exception, the exception and its stack trace
are wrapped in :class:`~pdart.rules.Result.Failure`.  Both are
subclasses of :class:`~pdart.rules.Result.Result`.

The end-user should never see these classes: they are only used
internally to implement functions like
:func:`~pdart.rules.Combinators.multiple_implementations`.
Results are wrapped to let Python work uniformly on all cases, then
the final result is unwrapped before presenting it to the user.
"""
import abc

import pdart.rules.ExceptionInfo

from typing import Any  # for mypy


class Result(object):
    """
    The result of running a function: either
    :class:`~pdart.rules.Result.Success` or
    :class:`~pdart.rules.Result.Failure`.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_success(self):
        # type: () -> bool
        """
        Return True if the Result is a
        :class:`~pdart.rules.Result.Success`.
        """
        pass

    def is_failure(self):
        # type: () -> bool
        """
        Return True if the Result is a
        :class:`~pdart.rules.Result.Failure`.
        """
        return not self.is_success()


class Failure(Result):
    """
    The result of running a function and failing: a wrapper around
    :class:`~pdart.rules.ExceptionInfo.ExceptionInfo`.
    """
    def __init__(self, exception_info):
        # type: (pdart.rules.ExceptionInfo.ExceptionInfo) -> None
        Result.__init__(self)
        self.exception_info = exception_info

    def is_success(self):
        # type: () -> bool
        return False

    def __str__(self):
        return '_Failure(%s)' % (self.exception_info, )


class Success(Result):
    """
    The result of running a function successfully: a wrapper around
    the returned value.
    """
    def __init__(self, value):
        # type: (Any) -> None
        Result.__init__(self)
        self.value = value

    def is_success(self):
        # type: () -> bool
        return True

    def __str__(self):
        return 'Success(%s)' % (self.value, )
