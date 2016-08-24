"""
The pdart.pds4labels package contains the code to build the XML needed
for PDS4 labels.

**New to PDART?** This package makes heavy use of XML templating,
whose implementation is found in the :mod:`pdart.xml` package.  You
might want to become familiar with templating first.

This package currently (as of 2016-08-24) contains paired
implementations of most of its functionality, one building labels
while walking the archive hierarchy and using the
:mod:`pdart.reductions` package, and the other drawing the information
from a SQLite database (tables defined in
:mod:`pdart.db.CreateDatabase`).  The ``reductions`` implementation
will probably be removed in the near future, so ignore it.

For, say, bundle labels, the ``reductions`` implementation is in
``BundleLabelReduction.py``.  The ``db`` implementation is in
``BundleLabelDB.py``.  Common code (mostly XML templates) is found in
``BundleLabelXml.py``.  ``BundleLabel.py`` simply includes both
implementations so we could import a single module to get either.
Other functionality follows the same file-naming pattern.
"""
