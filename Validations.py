import os.path
import pyfits
import re

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
            print 'Saw %d files.' % self.fileCount
            self.fileCount = None

    def doProductFile(self, product, file):
        self.fileCount += 1
        if self.fileCount % 200 == 0:
            print 'Saw %d files.' % self.fileCount


class CollectionUsesOneSingleInstrument(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.collectionInstruments = None

    def doBundle(self, bundle, before):
        if before:
            self.collectionInstruments = set()
        else:
            # Assert that for any bundle (equivalently, for any
            # proposal), all its collections use the same instrument
            if len(self.collectionInstruments) > 1:
                print 'UNEXPECTED: collections used %s' % \
                    str(self.collectionInstruments)
            self.collectionInstruments = None

    def doCollection(self, collection, before):
        if before:
            self.collectionInstruments.add(collection.instrument())


class ProductFilesHaveCollectionSuffix(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.collectionSuffix = None

    def doCollection(self, collection, before):
        if before:
            self.collectionSuffix = collection.suffix()
        else:
            self.collectionSuffix = None

    def doProductFile(self, product, file):
        # get file suffix
        fileBasename = os.path.basename(file)
        fileSuffix = re.match('\A[^_]+_([^\.]+)\..*\Z',
                              fileBasename).group(1)
        assert self.collectionSuffix == fileSuffix, \
            (product, file, self.collectionSuffix, fileSuffix)


class ProductFilesHaveBundleProposalId(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.bundleProposalId = None

    def doBundle(self, bundle, before):
        if before:
            self.bundleProposalId = bundle.proposalId()
        else:
            self.bundleProposalId = None

    def doProductFile(self, product, file):
        try:
            proposId = pyfits.getval(file, 'PROPOSID')
        except IOError:
            # TODO Put a good message here
            proposId = None
        except KeyError:
            proposId = None

        # if it exists, ensure it matches the bundleProposalId
        if proposId is not None:
            assert proposId == self.bundleProposalId


class ProductFilesHaveProductVisit(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.productVisit = None

    def doProduct(self, product, before):
        if before:
            self.productVisit = product.visit()
        else:
            self.productVisit = None

    def doProductFile(self, product, file):
        hstFile = HstFilename.HstFilename(file)
        assert hstFile.visit() == self.productVisit


class BundleContainsOneSingleHstInternalProposalId(Validation.NullValidation):
    def __init__(self):
        Validation.NullValidation.__init__(self)
        self.hstInternalProposalIds = None
        self.bundleProposalId = None

    def doProductFile(self, product, file):
        hstFile = HstFilename.HstFilename(file)
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
                assert len(self.hstInternalProposalIds) == 1, \
                    (bundle, self.hstInternalProposalIds)

            self.hstInternalProposalIds = None
            self.bundleProposalId = None

stdValidation = Validation.CompositeValidation([
        CountFilesValidation(),  # not really a validation
        ProductFilesHaveBundleProposalId(),
        ProductFilesHaveCollectionSuffix(),
        ProductFilesHaveProductVisit(),
        BundleContainsOneSingleHstInternalProposalId(),

        # This one doesn't seem to be true
        CollectionUsesOneSingleInstrument()

        # More to do?
        ])
