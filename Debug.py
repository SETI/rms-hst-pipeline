import functools
import pprint
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
            (_indent(), f.__name__,
             pprint.pformat(args), pprint.pformat(kwargs))
        _indent_level += 1
        try:
            res = f(*args, **kwargs)
            print '%sExiting %s()returning %s' % \
                (_indent(), f.__name__,
                 pprint.pformat(res))
            return res
        except BaseException as e:
            print '%sExiting %s() with exception %s' % \
                (_indent(), f.__name__,
                 pprint.pformat(e))
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
