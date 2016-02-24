"""
Module docstring goes here, with an explanation of why one needs to
use Heuristic objects instead of just calling functions.
"""
import abc


class Result(object):
    """
    The result of a call of Heuristic.run().  This is an abstract base
    class with only two subclasses: Success and Failure.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isSuccess(self):
        """Return True iff this is a Success instance."""
        pass


class Success(Result):
    """
    A result of a successful call to Heuristic.run().  Contains the
    result value.
    """
    def __init__(self, value):
        self.value = value

    def isSuccess(self):
        return True


class Failure(Result):
    """
    A result of a failed call of Heuristic.run().  Contains the
    exceptions raised by the calculation.
    """
    def __init__(self, exceptions):
        self.exceptions = exceptions

    def isSuccess(self):
        return False


class Heuristic(object):
    """
    A composable representation of a calculation on one argument that
    may fail.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self, arg):
        """
        Run the heuristic returning either a Success wrapper around
        the result, or a Failure wrapper around the raised exceptions.
        """
        pass


class HFunction(Heuristic):
    """
    A wrapper around a unary function to make it act as a Heuristic.
    """
    def __init__(self, func):
        """Create a Heuristic from the given function."""
        self.func = func

    def run(self, arg):
        """
        Run the wrapped function, returning a Result containing either
        the successful result value or the exceptions raised by the
        failure.
        """
        try:
            res = self.func(arg)
        except Exception as e:
            return Failure([e])
        else:
            return Success(res)


class HOrElses(Heuristic):
    """
    A composition of alternate Heuristics to calculate the same value
    different ways.

    If any succeeds, the composite succeeds.  If all fail, the
    composite fails.
    """

    def __init__(self, label, orElses):
        """
        Create a Heuristic from an optional label and a list of the
        alternative Heuristics to try.
        """
        self.label = label
        self.orElses = orElses

    def run(self, arg):
        """
        Run the alternative heuristics until one succeeds and return
        that value as a Success.  If none of them succeed, collect all
        their exceptions and return them as a Failure, labeling the
        group if the object has a label.
        """
        exceptions = []
        for h in self.orElses:
            res = h.run(arg)
            if res.isSuccess():
                return res
            else:
                exceptions.extend(res.exceptions)

        # if you made it here, all of them failed
        if self.label:
            return Failure([(self.label, exceptions)])
        else:
            return Failure(exceptions)


class HAndThens(Heuristic):
    """
    A composition of Heuristics to be chained, each one's result
    becoming the input to the next.

    If all succeed, the composite succeeds.  If any fails, the
    composite fails.
    """

    def __init__(self, label, andThens):
        """
        Create a Heuristic from an optional label and a list of
        Heuristics to be chained.
        """
        self.label = label
        self.andThens = andThens

    def run(self, arg):
        """
        Run the Heuristics chained, with the successful result value
        of each becoming the input argument to the next, and return
        that value as a Success.  If any of them fail, immediately
        abandon the rest of the calculation and return the raised
        exceptions as a Faillure, labeling the group if the object has
        a label.
        """
        if self.andThens:
            for h in self.andThens:
                res = h.run(arg)
                if res.isSuccess():
                    arg = res.value
                else:
                    if self.label:
                        return [(self.label, res.exceptions)]
                    else:
                        res
        else:
            res = Success(arg)

        # if you made it here, all of them succeeded
        return res


def _toHeuristic(arg):
    """Wrap the argument if necessary and return an instance of Heuristic."""
    if isinstance(arg, Heuristic):
        return arg
    elif hasattr(arg, '__call__'):
        # it's a function, or something like one
        return HFunction(arg)
    else:
        raise ValueError('argument was neither a Heuristic nor callable: %r' %
                         arg)


def hOrElses(*args):
    """
    Create an HOrElses using some syntactic sugar.  If the first
    argument is a string, it serves as a label; if not, the label is
    taken to be None.

    All other arguments become the components of the HOrElses.  The
    other arguments should all be Heuristics, but you may include
    unary functions, which will be wrapped to become HFunctions.
    """

    # Look for initial label
    if args and isinstance(args[0], str):
        label = args[0]
        args = args[1:]
    else:
        label = None

    return HOrElses(label, [_toHeuristic(arg) for arg in args])


def hAndThens(*args):
    """
    Create an HAndThens using some syntactic sugar.  If the first
    argument is a string, it serves as a label; if not, the label is
    taken to be None.

    All other arguments become the components of the HAndThens.  The
    other arguments should all be Heuristics, but you may include
    unary functions, which will be wrapped to become HFunctions.
    """

    # Look for initial label
    if args and isinstance(args[0], str):
        label = args[0]
        args = args[1:]
    else:
        label = None

    return HAndThens(label, [_toHeuristic(arg) for arg in args])
