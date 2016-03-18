import functools
import sys
import traceback

_indent_level = 0


def _indent():
    return '\n' + _indent_level * '> '


def trace(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        global _indent_level
        print '%sEntering %s(args=%s, kwargs=%s)' % \
            (_indent(), f.__name__, args, kwargs)
        _indent_level += 1
        try:
            res = f(*args, **kwargs)
            print '%sExiting %s(args=%s, kwargs=%s)\nreturning %r' % \
                (_indent(), f.__name__, args, kwargs, res)
            return res
        except BaseException as e:
            print '%sExiting %s(args=%s, kwargs=%s)\nwith exception %s' % \
                (_indent(), f.__name__, args, kwargs, e)
            raise
        finally:
            _indent_level -= 1
    return wrapped


def really_assert(condition, msg=None):
    if not condition:
        if msg:
            print msg
        else:
            print 'really_assert failed'
        traceback.print_stack()
        sys.exit(1)
