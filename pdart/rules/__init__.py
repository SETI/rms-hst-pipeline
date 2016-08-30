"""
The scripting language Python excels at letting developers
interactively program.  However, the PDART project requirements
include long-running programs handling terabytes of data: this
precludes working effectively interactively.

To work around this, we need to make Python smarter.  We teach Python
how to try different implementations of a function to try to find
something that works (sometimes called *rule-based programming*), and
we augment the error-reporting mechanism to report *multiple* errors
with their stack traces instead of just bailing out at the first.
This lets the developer fix multiple problems at the end of a failed
run, making her or him more effective.

The :mod:`pdart.rules` package contains the magic to make this
happen.

**New to PDART?** The meat of this package is the function
:func:`pdart.rules.Combinators.multiple_implementations`, used to
make *rules* (functions with multiple implementations).  Rules may be
used to try various heuristics to calculate a desired result.
"""
