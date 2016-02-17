import os.path
import re
import sys

import pyfits

import HstFilename
import Validation


class CountFilesValidation(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.fileCount = None

    def doArchive(self, archive, before):
        if before:
            self.fileCount = 0
        else:
            print >> sys.stderr, 'Saw %d files.' % self.fileCount
            self.fileCount = None

    def doProductFile(self, file):
        self.fileCount += 1
        if self.fileCount % 200 == 0:
            print >> sys.stderr, 'Saw %d files.' % self.fileCount


class ProductFilesHaveCollectionSuffix(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.collectionSuffix = None

    def doCollection(self, collection, before):
        if before:
            self.collectionSuffix = collection.suffix()
        else:
            self.collectionSuffix = None

    def doProductFile(self, file):
        # get file suffix
        fileBasename = file.basename
        fileSuffix = re.match('\A[^_]+_([^\.]+)\..*\Z',
                              fileBasename).group(1)
        self.assertEquals(self.collectionSuffix, fileSuffix,
                          'Unexpected suffix for file %s' % repr(file))


class ProductFilesHaveBundleProposalId(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.bundleProposalId = None

    def doBundle(self, bundle, before):
        if before:
            self.bundleProposalId = bundle.proposalId()
        else:
            self.bundleProposalId = None

    def doProductFile(self, file):
        try:
            proposId = pyfits.getval(file.fullFilepath(), 'PROPOSID')
        except IOError as e:
            # We know that much (all?) of the contents of hst_00000
            # are there due to IOErrors, so let's not report them.
            # Report others, though.
            if self.bundleProposalId != 0:
                self.report('IOError %s reading file %s of product %s' %
                            (e, file, file.component))
            proposId = None
        except KeyError:
            proposId = None

        # if it exists, ensure it matches the bundleProposalId
        if proposId is not None:
            self.assertEquals(self.bundleProposalId, proposId)


class ProductFilesHaveProductVisit(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.productVisit = None

    def doProduct(self, product, before):
        if before:
            self.productVisit = product.visit()
        else:
            self.productVisit = None

    def doProductFile(self, file):
        hstFile = HstFilename.HstFilename(file.fullFilepath())
        self.assertEquals(self.productVisit, hstFile.visit(),
                          'Unexpected visit value for file %s' % repr(file))


class BundleContainsOneSingleHstInternalProposalId(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.hstInternalProposalIds = None
        self.bundleProposalId = None

    def doProductFile(self, file):
        hstFile = HstFilename.HstFilename(file.fullFilepath())
        self.hstInternalProposalIds.add(hstFile.hstInternalProposalId())

    def doBundle(self, bundle, before):
        if before:
            self.bundleProposalId = bundle.proposalId()
            self.hstInternalProposalIds = set()
        else:
            # TODO It seems that hst_00000 is a grab bag of
            # lost files.  This needs to be fixed.
            # Otherwise...
            if self.bundleProposalId != 0:
                # Assert that for any bundle, all of its
                # files belong to the same project, using the
                # HST internal proposal ID codes.
                self.assertEquals(1, len(self.hstInternalProposalIds),
                                  'In bundle %s, saw HST proposal ids %s.' %
                                  (bundle, list(self.hstInternalProposalIds)))

            self.hstInternalProposalIds = None
            self.bundleProposalId = None

stdValidation = Validation.CompositeValidation([
        CountFilesValidation(),  # not really a validation
        ProductFilesHaveBundleProposalId(),
        ProductFilesHaveCollectionSuffix(),
        ProductFilesHaveProductVisit(),
        BundleContainsOneSingleHstInternalProposalId(),
        # More to do?
        ])
