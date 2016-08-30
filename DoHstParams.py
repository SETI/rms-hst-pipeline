"""
**SCRIPT:** Run through the archive and for (only) the first product
in the first RAW data collection, calculate its label and print it
prettily.
"""
import subprocess

from pdart.pds4.Archives import *
from pdart.pds4labels.ProductLabel import *
from pdart.rules.Combinators import *


def make_one_product_label():
    """Prints a product label for the first RAW product."""
    arch = get_any_archive()
    for b in arch.bundles():
        for c in b.collections():
            if c.prefix() == 'data' and c.suffix() == 'raw':
                for p in c.products():
                    def thunk():
                        stdin = make_product_label(p, True)
                        sp = subprocess.Popen(['xmllint', '--format', '-'],
                                              stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
                        stdout, stderr = sp.communicate(stdin)
                        exit_code = sp.returncode
                        if stdout:
                            print stdout
                        if stderr:
                            print stderr
                        if exit_code:
                            print exit_code
                    raise_verbosely(thunk)
                    return

if __name__ == '__main__':
    make_one_product_label()
