"""
**SCRIPT:** Finds the first product in the archive whose FITS file is
parseable, generates its ``<hst:HST />`` XML element and prints it.
"""
from pdart.pds4.Archives import *
from pdart.pds4labels.HstParametersReduction import *
from pdart.reductions.Reduction import *
from pdart.rules.Combinators import *


def get_product():
    # type: () -> Product
    """
    Return the first product in the archive whose FITS file is
    parseable
    """
    arch = get_any_archive()
    for b in arch.bundles():
        for c in b.collections():
            for p in c.products():
                for f in p.files():
                    try:
                        filepath = f.full_filepath()
                        fits = pyfits.open(filepath)
                        fits.close()
                        return p
                    except IOError:
                        pass


def run():
    # type: () -> None
    reduction = HstParametersReduction()
    runner = DefaultReductionRunner()
    p = get_product()
    print raise_verbosely(lambda: runner.run_product(reduction, p))


if __name__ == '__main__':
    run()
