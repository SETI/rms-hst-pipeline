import traceback

from pdart.exceptions.ExceptionInfo import *
from pdart.exceptions.Result import Failure, Success


def _code_to_rcode(func):
    def rfunc(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
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
    the original raises an Exception, the result function will instead
    raise a CalculationException containing ExceptionInfo for the
    exception.
    """
    return _rcode_to_code(_code_to_rcode(func))


def multiple_implementations(label, *funcs):
    """
    Given a string label and a number of functions, return the result
    of the first function that succeeds or raise a
    CalculationException containing GroupedExceptionInfo for the
    exceptions raised by each function.

    This is a generalization of function call to multiple, alternative
    implementations.  If any one succeeds, you get the result.  If
    they all fail, you get all the exceptions and all the stack traces
    wrapped into a GroupedExceptionInfo in a CalculationException.
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
