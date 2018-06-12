import os
import sys

import fs.path

from pdart.pds4.HstFilename import HstFilename

_ROOT = '/Volumes/PDART-5TB Part Deux/bulk-download'

def xlate_filenames():
    for (dirpath, dirnames, filenames) in os.walk(_ROOT):
        if len(filenames) > 0:
            path = fs.path.iteratepath(dirpath)
            depth = len(path)
            assert depth == 7
            _, _, _, hst, _, _, hst_name = path
            hst_name = hst_name.lower()
            for filename in filenames:
                _, ext = fs.path.splitext(filename)
                assert ext == '.fits'
                hst_filename = HstFilename(filename)

                coll = 'data_%s_%s' % (hst_filename.instrument_name(), 
                                       hst_filename.suffix())
                new_path = fs.path.join(hst, coll, hst_name, filename)
                print new_path

if __name__ == '__main__':
    xlate_filenames()
