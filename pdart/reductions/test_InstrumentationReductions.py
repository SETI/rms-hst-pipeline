from pdart.reductions.InstrumentationReductions import *
from pdart.reductions.Reduction import *


def test_InstrumentationReductions():
    """Just a smoketest to force parsing of InstrumentationReductions."""
    lbr = LogBundlesReduction()
    assert lbr is not None
