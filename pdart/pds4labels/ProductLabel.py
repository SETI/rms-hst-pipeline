"""
Functionality to build a product label.  Imports two implementations:
the original :class:`~pdart.reductions.Reduction.Reduction`
implementation and the new SQLite implementation.  (The former is
probably obsolete.)
"""
from pdart.pds4labels.ProductLabelDB import *
from pdart.pds4labels.ProductLabelReduction import *
