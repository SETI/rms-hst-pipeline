"""
SCRIPT: Run through the archive and generate labels, browse products,
and their labels.  Do not validate.  If it fails at any point, print
the combined exception as XML to stdout.
"""
from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4labels.BrowseProductImage import *
from pdart.pds4labels.BrowseProductLabel import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *
from pdart.reductions.CompositeReduction import *
from pdart.reductions.InstrumentationReductions import *

import pdart.add_pds_tools
import picmaker


class MakeRawBrowseReduction(CompositeReduction):
    """
    When run on an archive, create browse collections for each RAW
    collection.
    """
    def __init__(self):
        CompositeReduction.__init__(self,
                                    [BrowseProductImageReduction(),
                                     BrowseProductLabelReduction()])


class MakeLabelsReduction(CompositeReduction):
    def __init__(self):
        CompositeReduction.__init__(self,
                                    [BundleLabelReduction(),
                                     CollectionLabelReduction(),
                                     ProductLabelReduction()])

if __name__ == '__main__':
    archive = get_any_archive()
    reduction = CompositeReduction([LogProductsReduction(),
                                    MakeLabelsReduction(),
                                    MakeRawBrowseReduction()])
    raise_verbosely(lambda: run_reduction(reduction, archive))