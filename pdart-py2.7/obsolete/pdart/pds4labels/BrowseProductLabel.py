"""
Functionality to build a raw browse product label.  Imports two
implementations: the original
:class:`~pdart.reductions.Reduction.Reduction` implementation and the
new SQLite implementation.  (The former is probably obsolete.)
"""
from pdart.pds4labels.BrowseProductLabelDB import *
from pdart.pds4labels.BrowseProductLabelReduction import *
