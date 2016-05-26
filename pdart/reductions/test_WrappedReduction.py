from pdart.reductions.WrappedReduction import *
from pdart.reductions.test_Reduction import *


def test_check_wrapped_reduction():
    arch = get_any_archive()
    for red in [CheckBundleReduction(),
                CheckCollectionReduction(),
                CheckProductReduction(),
                CheckFileReduction()]:
        run_reduction(WrappedReduction(red), arch)
