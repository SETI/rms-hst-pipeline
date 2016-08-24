"""
Functionality to build a bundle label.  Imports two implementations:
the original :class:`~pdart.reductions.Reduction.Reduction`
implementation and the new SQLite implementation.  (The former is
probably obsolete.)
"""
from pdart.pds4labels.BundleLabelDB import *
from pdart.pds4labels.BundleLabelReduction import *
