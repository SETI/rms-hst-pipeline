import pdart.reductions.BadFitsFileReduction
from pdart.reductions.Reduction import *


def test_BadFitsFileReduction():
    """Just a smoketest to force parsing of BadFitsFileReduction."""
    bffr = pdart.reductions.BadFitsFileReduction.BadFitsFileReduction(
        Reduction())
    assert bffr is not None
