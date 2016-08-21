import traceback

from pdart.exceptions.ExceptionInfo import *
from pdart.exceptions.Result import Failure, Success


def _code_to_rcode(func):
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
    the original raises an :class:`Exception`, the result function
    will instead raise a :class:`CalculationException` containing
    :class:`ExceptionInfo` for the exception.
    """
    return _rcode_to_code(_code_to_rcode(func))


def multiple_implementations(label, *funcs):
    """
    Given a string label and a number of functions, return the result
    of the first function that succeeds or raise a
    :class:`CalculationException` containing
    :class:`GroupedExceptionInfo` for the exceptions raised by each
    function.

    This is a generalization of function call to calling multiple
    alternative implementations.  If any one succeeds, you get the
    result.  If they all fail, you get all the exceptions and all the
    stack traces wrapped into a :class:`GroupedExceptionInfo` in a
    :class:`CalculationException`.
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
    fails, raise a :class:`CalculationException` containing all the
    exceptions and stack traces in a :class:`GroupedExceptionInfo`.
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
    CalculationException, pretty-print the full exception info as XML
    and reraise the exception.
    """
    try:
        return thunk()
    except CalculationException as ce:
        print ce.exception_info.to_pretty_xml()
        raise
