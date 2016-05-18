"""
SCRIPT: Run through the archive and generate a browse collection for
each RAW collection, writing them to disk including the collection
inventory and verified label.  If it fails at any point, print the
combined exception as XML to stdout.  Optionally delete the
collections afterwards.  (Does not currently create product labels.)
"""

from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4labels.BrowseProductImage import *
from pdart.pds4labels.BrowseProductLabel import *
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


if __name__ == '__main__':
    archive = get_any_archive()
    reduction = CompositeReduction([LogCollectionsReduction(),
                                    MakeRawBrowseReduction()])
    raise_verbosely(lambda: run_reduction(reduction, archive))
