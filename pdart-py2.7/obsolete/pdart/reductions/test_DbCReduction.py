import pdart.reductions.DbCReduction
from pdart.reductions.Reduction import *


def test_DbCReduction():
    """Just a smoketest to force parsing of DbCReduction."""
    dbcr = pdart.reductions.DbCReduction.DbCReduction()
    assert dbcr is not None
