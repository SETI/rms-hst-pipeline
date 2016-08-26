from pdart.reductions.DbCReduction import *
from pdart.reductions.Reduction import *


def test_DbCReduction():
    """Just a smoketest to force parsing of DbCReduction."""
    dbcr = DbCReduction()
    assert dbcr is not None
