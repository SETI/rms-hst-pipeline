"""
**SCRIPT:** Run through the archive and generate labels (but not
browse products or their labels).  Possibly validate them.  If it
fails at any point, print the combined exception as XML to stdout.
"""
import sys

from pdart.pds4.Archives import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.InstrumentationReductions import *
from pdart.rules.Combinators import *


class _ProductLabelReductionWithMessage(ProductLabelReduction):
    """
    Summarizes a product into its label (or ``None`` if the FITS file
    cannot be read, noting the problem).
    """
    def __init__(self, verify):
        ProductLabelReduction.__init__(self, verify)

    def reduce_product(self, archive, lid, get_reduced_fits_files):
        res = ProductLabelReduction.reduce_product(self, archive, lid,
                                                   get_reduced_fits_files)
        if res is None:
            note_problem(lid.product_id, 'bad_fits_file')
        return res


class _MakeLabelsReduction(CompositeReduction):
    """
    Summarizes the archive into ``None``, generating labels (and
    possibly validating them) as a side-effect.
    """
    def __init__(self, verify):
        CompositeReduction.__init__(self,
                                    [BundleLabelReduction(verify),
                                     CollectionLabelReduction(verify),
                                     _ProductLabelReductionWithMessage(verify)
                                     ])

if __name__ == '__main__':
    VERIFY = True

    archive = get_any_archive()
    reduction = CompositeReduction([LogProductsReduction(),
                                    _MakeLabelsReduction(VERIFY)])
    raise_verbosely(lambda: run_reduction(reduction, archive))
