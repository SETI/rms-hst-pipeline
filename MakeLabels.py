"""
SCRIPT: Run through the archive and generate labels (but not browse
products or labels).  Do not validate them.  If it fails at any point,
print the combined exception as XML to stdout.
"""
from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.InstrumentationReductions import *


class MakeLabelsReduction(CompositeReduction):
    def __init__(self, verify):
        CompositeReduction.__init__(self,
                                    [BundleLabelReduction(verify),
                                     CollectionLabelReduction(verify),
                                     ProductLabelReduction(verify)])

if __name__ == '__main__':
    archive = get_any_archive()
    verify = True
    reduction = CompositeReduction([LogProductsReduction(),
                                    MakeLabelsReduction(verify)])
    raise_verbosely(lambda: run_reduction(reduction, archive))
