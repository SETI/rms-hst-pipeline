"""
A pass to collect data from the FITS headers in an archive.
"""
import collections
import pprint
import sys

import pyfits

import FileArchives
import Pass
import Targets


class FitsPass(Pass.NullPass):
    def __init__(self):
        super(Pass.NullPass, self).__init__()

    def do_product_file(self, file):
        try:
            fits = pyfits.open(file.full_filepath())
            header = fits[0].header
            try:
                targname = header['targname']
                target = Targets.targname_to_target(targname)
                if target:
                    print 'In %s, target %s' % (file, target)
                else:
                    pass
                    # print 'In %s, unknown target %s' % (file, targname)
            except KeyError:
                pass
                # print 'No key TARGNAME'
            fits.close()
        except IOError as e:
            # self.fits_error = True
            # print 'Exception on %s: %s' % (file, e)
            pass

if __name__ == '__main__':
    a = FileArchives.get_mini_archive()
    f = FitsPass()
    r = Pass.PassRunner()
    r.run(a, f)
    # pp = pprint.PrettyPrinter(indent=4, width=50)
    # pp.pprint(f.targ_dict)
