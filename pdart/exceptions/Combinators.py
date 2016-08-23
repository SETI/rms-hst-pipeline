"""
This module provides tools for building new functions by combining
functions.  (Functions that combine functions are called
"combinators."  I usually accent the first syllable.)

There are two main combinators here. The first is
:func:`multiple_implementations` which is used to create a function
that tries multiple ways to get a result.  It's a big **or**: if any
of the implementations work, it works.  The other is
:func:`parallel_list` which acts as a big **and**: all the component
functions must work or it does not work.

The utility function :func:`raise_verbosely` pretty-prints the
exception(s) as XML if the thunk it's called on fails.  This is needed
for a compound exception to be legible and is used at the top-level of
a script.

**You don't need to understand the internals of**
:func:`multiple_implementations` or :func:`parallel_list` **to use
them.**

The key concept used here is converting from "code" (normal Python
functions that either return values or raise exceptions) to "rcode" (a
function that uniformly returns a :class:`~pdart.exception.Result`
value and *never* raises an exception), or the other way (from code
that returns a wrapped :class:`~pdart.exception.Result` into normal
Python code that returns an unwrapped result or raises an exception).

Our normal process to combine multiple Python functions into a single
Python function is to first convert them all into *rcode*, so Python
can look at both normal and exception results uniformly.  We then
create a function that handles the results from the *rcode*: since we
are guaranteed that no exceptions will be thrown (they are all
converted into :class:`~pdart.exception.Failure` s), Python can work on
the result like it can on any data.  Then we convert this *rcode* back
into normal code.  The end-user will never see the *rcode*.

"""
import traceback

from pdart.exceptions.ExceptionInfo import *
from pdart.exceptions.Result import Failure, Success


def _code_to_rcode(func):
    """
    Convert from a function that either returns a value or raises an
    exception (i.e., a normal Python function) to a function that
    always returns a :class:`~pdart.exception.Result`.
    """
    def rfunc(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except CalculationException as ce:
            return Failure(ce.exception_info)
        except Exception as e:
            exception_info = SingleExceptionInfo(e, traceback.format_exc())
            return Failure(exception_info)
        return Success(res)
    return rfunc


def _rcode_to_code(rfunc):
    """
    Convert from a function that returns a
    :class:`~pdart.exception.Result` to a function that either returns
    an unwrapped value or raises a
    :exc:`~pdart.exception.CalculationException`.
    """
    def func(*args, **kwargs):
        res = rfunc(*args, **kwargs)
        if res.is_success():
            return res.value
        else:
            raise CalculationException(
                '', res.exception_info)
    return func


def normalized_exceptions(func):
    """
    Given a function, return an equivalent function except that when
    the original raises an :class:`~pdart.exception.Exception`, the
    result function will instead raise a
    :exc:`~pdart.exception.CalculationException` containing
    :class:`~pdart.exception.ExceptionInfo` for the exception.
    """
    return _rcode_to_code(_code_to_rcode(func))


def _create_joined_documentation(funcs):
    """
    Synthesize and return a docstring for a combination of functions
    created by :func:`~pdart.exceptions.multiple_implementations`.
    """
    name_docs = ["%s: %s" % (func.__name__, func.__doc__) for func in funcs]
    return ("has more than one implementations:\n" + '\n'.join(name_docs))


def _create_joined_name(label, funcs):
    """
    Synthesize and return a name for a function created by
    :func:`~pdart.exceptions.multiple_implementations`.
    """
    names = [func.__name__ for func in funcs]
    return "multiple_implementations(%r, %s)" % (label, ', '.join(names))


def multiple_implementations(label, *funcs):
    """
    Given a string label and a number of functions, return the result
    of the first function that succeeds or raise a
    :exc:`~pdart.exception.CalculationException` containing
    :class:`~pdart.exception.GroupedExceptionInfo` for the exceptions
    raised by each function.

    This is a generalization of function call to calling multiple
    alternative implementations.  If any one succeeds, you get the
    result.  If they all fail, you get all the exceptions and all the
    stack traces wrapped into a
    :class:`~pdart.exception.GroupedExceptionInfo` in a
    :exc:`~pdart.exception.CalculationException`.
    """
    def afunc(*args, **kwargs):
        exception_infos = []
        for func in funcs:
            res = _code_to_rcode(func)(*args, **kwargs)
            if res.is_success():
                return res.value
            else:
                exception_infos.append(res.exception_info)
        # if we got here, there were no successes
        exception_info = GroupedExceptionInfo(label, exception_infos)
        raise CalculationException(label, exception_info)

    afunc.__name__ = _create_joined_name(label, funcs)
    afunc.__doc__ = _create_joined_documentation(funcs)

    return afunc


# def parallel_arguments(label, func, *arg_funcs):
#     def pfunc():
#         exception_infos = []
#         results = []
#         for arg_func in arg_funcs:
#             arg_res = _code_to_rcode(arg_func)()
#             if arg_res.is_success():
#                 results.append(arg_res.value)
#             else:
#                 exception_infos.append(arg_res.exception_info)
#         if exception_infos:
#             # We failed if any arg_func failed
#             exception_info = GroupedExceptionInfo(label, exception_infos)
#             raise CalculationException(exception_info)
#         else:
#             return f(results)
#     return pfunc

def parallel_list(label, arg_funcs):
    """
    Given a string label and a list of functions that take no
    arguments (thunks), run the functions in parallel, and if all
    succeed, return the list of the results.  If any one or more
    fails, raise a :exc:`~pdart.exception.CalculationException`
    containing all the exceptions and stack traces in a
    :class:`~pdart.exception.GroupedExceptionInfo`.
    """

    # Style note: unlike multiple_interpretations, this takes a list
    # rather than a *list because it explicitly says "list": you're
    # creating a parallel list from a plist.  That's why.

    exception_infos = []
    results = []
    for arg_func in arg_funcs:
        arg_res = _code_to_rcode(arg_func)()
        if arg_res.is_success():
            results.append(arg_res.value)
        else:
            exception_infos.append(arg_res.exception_info)
    if exception_infos:
        # We failed if any arg_func failed
        exception_info = GroupedExceptionInfo(label, exception_infos)
        raise CalculationException(label, exception_info)
    else:
        return results


def raise_verbosely(thunk):
    """
    Run the thunk, returning a result.  If it raises a
    :exc:`~pdart.exception.CalculationException`, pretty-print the
    full exception info as XML and reraise the exception.  For
    debugging.
    """
    try:
        return thunk()
    except CalculationException as ce:
        print ce.exception_info.to_pretty_xml()
        raise
