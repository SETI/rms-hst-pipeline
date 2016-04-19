import pdart.pds4.Archives

from pdart.exceptions.ExceptionInfo import CalculationException
from pdart.reductions.Reduction import *


class RecursiveReduction(Reduction):
    def reduce_archive(self, archive_root, get_reduced_bundles):
        return 1 + sum(get_reduced_bundles())

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        return 1 + sum(get_reduced_collections())

    def reduce_collection(self, archive, lid, get_reduced_products):
        return 1 + sum(get_reduced_products())

    def reduce_product(self, archive, lid, get_reduced_files):
        return 1 + sum(get_reduced_files())

    def reduce_fits(self, file, get_reduced_hdus):
        try:
            return 1 + sum(get_reduced_hdus())
        except Exception:
            return 1


def test_reductions():
    arch = pdart.pds4.Archives.get_any_archive()
    try:
        count = ReductionRunner().run_archive(RecursiveReduction(), arch)
        assert count == 137567
    except CalculationException as ce:
        print ce.exception_info.to_pretty_xml()
        raise
