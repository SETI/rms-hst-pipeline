"""
A pass to collect data from the FITS headers in an archive.
"""
import collections
import pprint
import sys

import pyfits

import FileArchives
import Pass


class FitsPass(Pass.NullPass):
    def __init__(self):
        self.fits_dict = None
        self.fits_error = False
        self.full_dict = {}
        super(Pass.NullPass, self).__init__()

    def do_product(self, product, before):
        if before:
            self.fits_dict = collections.defaultdict(set)
            self.fits_error = False
        else:
            if not self.fits_error:
                self.full_dict[product.lid.lid] = dict(self.fits_dict)
                # sys.exit(0)

    def do_product_file(self, file):
        try:
            fits = pyfits.open(file.full_filepath())
            header = fits[0].header
            for k, v in header.iteritems():
                self.fits_dict[k].add(v)
            fits.close()
        except Exception as e:
            self.fits_error = True
            # print 'Exception: %s' % e

if __name__ == '__main__':
    a = FileArchives.get_mini_archive()
    f = FitsPass()
    r = Pass.PassRunner()
    r.run(a, f)
    pp = pprint.PrettyPrinter(indent=4, width=50)
    pp.pprint(f.full_dict)
