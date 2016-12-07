import pdart.reductions.WrappedReduction
from pdart.reductions.test_Reduction import *


def test_check_wrapped_reduction():
    # type: () -> None
    arch = get_any_archive()
    for red in [CheckBundleReduction(),
                CheckCollectionReduction(),
                CheckProductReduction(),
                CheckFileReduction()]:
        run_reduction(pdart.reductions.WrappedReduction.WrappedReduction(red),
                      arch)
