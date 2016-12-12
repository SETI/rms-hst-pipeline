"""
**SCRIPT:** Run through the archive and print the sorted set of FITS
files which raise ``IOError`` s when you try to parse them.
"""

from pdart.pds4.Archives import *
from pdart.reductions.BadFitsFileReduction import *
from pdart.rules.Combinators import *

from typing import Iterable, Set, TypeVar  # for mypy
T = TypeVar('T')  # for mypy


def _unions(sets):
    # type: (Iterable[Set[T]]) -> Set[T]
    """Union a list of sets."""
    res = set()
    # type: Set[T]
    for s in sets:
        res |= s
    return res


class _CollectFilesReduction(Reduction):
    """
    Summarizes an archive into a set of reduced FITS files.
    """

    def reduce_archive(self, archive_root, get_reduced_bundles):
        return _unions(get_reduced_bundles())

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        return _unions(get_reduced_collections())

    def reduce_collection(self, archive, lid, get_reduced_products):
        return _unions(get_reduced_products())

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        return set([ff for ff in get_reduced_fits_files() if ff is not None])

    def reduce_fits_file(self, file, get_reduced_hdus):
        return None


class CheckBadFitsFileReduction(BadFitsFileReduction):
    """
    Summarizes an archive into a set of full filepaths of FITS files
    that cannot be opened.
    """
    def __init__(self):
        BadFitsFileReduction.__init__(self, _CollectFilesReduction())

    def bad_fits_file_reduction(self, file):
        return file.full_filepath()


if __name__ == '__main__':
    reduction = CheckBadFitsFileReduction()
    archive = get_any_archive()

    def thunk():
        s = run_reduction(reduction, archive)
        for f in sorted(list(s)):
            print f
    raise_verbosely(thunk)

# Bad files from the mini-archive are:
#
# Archive/hst_05167/data_wfpc2_cmh/visit_04/u2no0401j_cmh.fits
# Archive/hst_05167/data_wfpc2_cmi/visit_04/u2no0401j_cmi.fits
# Archive/hst_05167/data_wfpc2_cmj/visit_04/u2no0401j_cmj.fits
# Archive/hst_05215/data_wfpc2_cmh/visit_01/u2mi0101j_cmh.fits
# Archive/hst_05215/data_wfpc2_cmi/visit_01/u2mi0101j_cmi.fits
# Archive/hst_05215/data_wfpc2_cmj/visit_01/u2mi0101j_cmj.fits
# Archive/hst_05313/data_wfpc2_cmh/visit_01/u2mo0101j_cmh.fits
# Archive/hst_05313/data_wfpc2_cmi/visit_01/u2mo0101j_cmi.fits
# Archive/hst_05313/data_wfpc2_cmj/visit_01/u2mo0101j_cmj.fits
# Archive/hst_05361/data_wfpc2_cmh/visit_01/u2ke0101j_cmh.fits
# Archive/hst_05361/data_wfpc2_cmj/visit_01/u2ke0101j_cmj.fits
# Archive/hst_05489/data_wfpc2_cmh/visit_01/u2j70101j_cmh.fits
# Archive/hst_05489/data_wfpc2_cmi/visit_01/u2j70101j_cmi.fits
# Archive/hst_05489/data_wfpc2_cmj/visit_01/u2j70101j_cmj.fits
# Archive/hst_05776/data_wfpc2_cmh/visit_01/u2kr0101j_cmh.fits
# Archive/hst_05776/data_wfpc2_cmi/visit_01/u2kr0101j_cmi.fits
# Archive/hst_05782/data_wfpc2_cmh/visit_01/u2on0101j_cmh.fits
# Archive/hst_05782/data_wfpc2_cmi/visit_01/u2on0101j_cmi.fits
# Archive/hst_05782/data_wfpc2_cmj/visit_01/u2on0101j_cmj.fits
# Archive/hst_05783/data_wfpc2_cmh/visit_02/u2lw0201j_cmh.fits
# Archive/hst_05783/data_wfpc2_cmi/visit_02/u2lw0201j_cmi.fits
# Archive/hst_05783/data_wfpc2_cmj/visit_02/u2lw0201j_cmj.fits
# Archive/hst_05832/data_wfpc2_cmh/visit_51/u2q95101j_cmh.fits
# Archive/hst_05832/data_wfpc2_cmi/visit_51/u2q95101j_cmi.fits
# Archive/hst_05832/data_wfpc2_cmj/visit_51/u2q95101j_cmj.fits
# Archive/hst_05836/data_wfpc2_cmh/visit_01/u2tf0101j_cmh.fits
# Archive/hst_05836/data_wfpc2_cmi/visit_01/u2tf0101j_cmi.fits
# Archive/hst_05836/data_wfpc2_cmj/visit_01/u2tf0101j_cmj.fits
# Archive/hst_05837/data_wfpc2_cmh/visit_01/u2p60101j_cmh.fits
# Archive/hst_05837/data_wfpc2_cmi/visit_01/u2p60101j_cmi.fits
# Archive/hst_06028/data_wfpc2_cmh/visit_01/u2oz0101j_cmh.fits
# Archive/hst_06028/data_wfpc2_cmi/visit_01/u2oz0101j_cmi.fits
# Archive/hst_06028/data_wfpc2_cmj/visit_01/u2oz0101j_cmj.fits
# Archive/hst_06029/data_wfpc2_cmh/visit_03/u2q60301j_cmh.fits
# Archive/hst_06029/data_wfpc2_cmi/visit_03/u2q60301j_cmi.fits
# Archive/hst_06029/data_wfpc2_cmj/visit_03/u2q60301j_cmj.fits
# Archive/hst_06030/data_wfpc2_cmh/visit_01/u2qe0101j_cmh.fits
# Archive/hst_06030/data_wfpc2_cmi/visit_01/u2qe0101j_cmi.fits
# Archive/hst_06030/data_wfpc2_cmj/visit_01/u2qe0101j_cmj.fits
# Archive/hst_06141/data_wfpc2_cmh/visit_01/u2mu0101j_cmh.fits
# Archive/hst_06141/data_wfpc2_cmi/visit_01/u2mu0101j_cmi.fits
# Archive/hst_06141/data_wfpc2_cmj/visit_01/u2mu0101j_cmj.fits
# Archive/hst_06145/data_wfpc2_cmh/visit_02/u2n20201j_cmh.fits
# Archive/hst_06145/data_wfpc2_cmj/visit_02/u2n20201j_cmj.fits
# Archive/hst_06216/data_wfpc2_cmh/visit_01/u2oo0101j_cmh.fits
# Archive/hst_06216/data_wfpc2_cmi/visit_01/u2oo0101j_cmi.fits
# Archive/hst_06216/data_wfpc2_cmj/visit_01/u2oo0101j_cmj.fits
# Archive/hst_06218/data_wfpc2_cmh/visit_01/u2r60101j_cmh.fits
# Archive/hst_06218/data_wfpc2_cmi/visit_01/u2r60101j_cmi.fits
# Archive/hst_06218/data_wfpc2_cmj/visit_01/u2r60101j_cmj.fits
