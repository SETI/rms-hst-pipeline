"""
The pdart.reductions package provides tooling to remove the need for
boilerplate code when extracting data from the PDART archive.

**New to PDART?** The meat of this package is in
:mod:`pdart.reductions.Reduction` and you should go there if you need
to learn about reductions.  However, you probably won't need to write
any, as we usually start out with one big extraction (extracting data
from the archive into a SQLite database) and then work only with the
extracted data.

However, reductions can be very useful when writing quick scripts to
verify something about the archive.
"""
