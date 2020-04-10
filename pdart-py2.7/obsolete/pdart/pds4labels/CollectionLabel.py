"""
Functionality to build a collection label.  Imports two implementations:
the original :class:`~pdart.reductions.Reduction.Reduction`
implementation and the new SQLite implementation.  (The former is
probably obsolete.)
"""
from pdart.pds4labels.CollectionLabelDB import *
from pdart.pds4labels.CollectionLabelReduction import *
