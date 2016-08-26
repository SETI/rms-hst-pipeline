from pdart.reductions.BadFitsFileReduction import *
from pdart.reductions.Reduction import *


def test_BadFitsFileReduction():
    """Just a smoketest to force parsing of BadFitsFileReduction."""
    bffr = BadFitsFileReduction(Reduction())
    assert bffr is not None
