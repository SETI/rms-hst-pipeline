"""
**SCRIPT:** Run through the archive and print the sorted set of
collection suffixes.
"""
from pdart.pds4.Archives import *
from pdart.pds4.Collection import *
from pdart.reductions.Reduction import *
from pdart.rules.Combinators import *


def _unions(sets):
    """Union up a list of sets."""
    res = set()
    for s in sets:
        res |= s
    return res


class CheckSuffixesReduction(Reduction):
    """
    Summarizes an archive into a set of collection suffixes.
    """
    def reduce_archive(self, archive_root, get_reduced_bundles):
        return _unions(get_reduced_bundles())

    def reduce_bundle(self, archive, lid, get_reduced_collections):
        return set(get_reduced_collections())

    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        return collection.suffix()


if __name__ == '__main__':
    reduction = CheckSuffixesReduction()
    archive = get_any_archive()

    def thunk():
        s = run_reduction(reduction, archive)
        print sorted(list(s))
    raise_verbosely(thunk)

# Suffixes found in the archive are:
#
# 'asn', 'c0m', 'c1m', 'c3m', 'cmh', 'cmi', 'cmj', 'crj', 'd0m',
# 'drz', 'flt', 'flt_hlet', 'jif', 'jit', 'q0m', 'q1m', 'raw', 'shm',
# 'spt', 'trl', 'x0m'
#
# Association tables: 'asn'
#
# (Note possible conflicts between instruments here below.)
#
# Obslogs: 'c0m', 'c1m', 'c3m', 'cmh', 'cmi', 'cmj', 'jif', 'jit'
#
# Calibrated science data: 'c0m'
#
# Data quality for calibrated science data: 'c1m'
#
# Throughput table for obs mode: 'c3m'
#
# Association product: 'crj'
#
# Raw science data: 'd0m'
#
# Mosaics: 'drz'
#
# Single fully-calibrated MAMA image: 'flt'
#
# Data quality for raw science data: 'q0m'
#
# Data quality for extracted engineering data: 'q1m'
#
# Raw data: 'raw'
#
# Support files: 'spt'
#
# Trailer file containing calacs processing comments: 'trl'
#
# Remaining:  'crc', 'd0m',  'drc', 'flc',  'flt_hlet',  'ima', 'lrc',
# 'lsp', 'pdq', 'sfl', 'shm', 'x0m'
