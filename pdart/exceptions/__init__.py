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

The :mod:`pdart.exceptions` package contains the magic to make this
happen.
"""
