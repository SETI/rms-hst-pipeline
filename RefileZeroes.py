"""
A script to refile files whose project number was incorrectly taken as
zero, either because the FITS header did report that, or because
pyfits could not read the header.  We assume that HST consistently
encodes project numbers into filenames, and we use the global
set of files to match project numbers with lost files.
"""
import os
import os.path
import sys

import Bundle
import FileArchives
import HstFilename
import IdTables2
import LID
import Product


def basenameToProductLID(basename):
    hst = HstFilename.HstFilename(basename)
    projectId = IdTables2.charToIntDict[hst.hstInternalProposalId()]
    instrument = hst.instrumentName()
    suffix = hst.suffix()
    visit = hst.visit()
    return LID.LID('urn:nasa:pds:hst_%05d:data_%s_%s:visit_%s' %
                   (int(projectId), instrument, suffix, visit))


def doRefileZeroes(mkdirs=False, move=False):
    archive = FileArchives.getAnyArchive()
    lid = LID.LID('urn:nasa:pds:hst_00000')
    bundle = Bundle.Bundle(archive, lid)
    for product in bundle.products():
        for f in product.files():
            lid = basenameToProductLID(f.basename)
            dstProduct = Product.Product(archive, lid)
            dstDir = dstProduct.directoryFilepath()
            if os.path.isdir(dstDir):
                exists = 'exists'
            elif os.path.isfile(dstDir):
                exists = 'exists but is a file (?!)'
                sys.exit(1)
            else:
                exists = 'does not exist'
                if mkdirs:
                    os.makedirs(dstDir)
                    exists += ' but was created'
            print dstProduct, exists
            srcFile = f.fullFilepath()
            dstFile = os.path.join(dstDir, f.basename)
            print '%s => %s' % (srcFile, dstFile)
            if move:
                os.rename(srcFile, dstFile)
                print 'Moved %s => %s' % (srcFile, dstFile)

if __name__ == '__main__':
    doRefileZeroes()
