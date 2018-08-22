from pdart.reductions.InstrumentationReductions import *


def test_InstrumentationReductions():
    """Just a smoketest to force parsing of InstrumentationReductions."""
    lbr = LogBundlesReduction()
    assert lbr is not None
