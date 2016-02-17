import os.path
import re
import sys

import pyfits

import HstFilename
import Pass


class CountFilesPass(Pass.NullPass):
    def __init__(self):
        self.fileCount = None
        Pass.NullPass.__init__(self)

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


class ProductFilesHaveCollectionSuffix(Pass.NullPass):
    def __init__(self):
        self.collectionSuffix = None
        Pass.NullPass.__init__(self)

    def doCollection(self, collection, before):
        if before:
            self.collectionSuffix = collection.suffix()
        else:
            self.collectionSuffix = None

    def doProductFile(self, file):
        # get file suffix
        fileSuffix = HstFilename.HstFilename(file.basename).suffix()
        self.assertEquals(self.collectionSuffix, fileSuffix,
                          'Unexpected suffix for file %s' % repr(file))


class ProductFilesHaveBundleProposalId(Pass.NullPass):
    def __init__(self):
        self.bundleProposalId = None
        Pass.NullPass.__init__(self)

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


class ProductFilesHaveProductVisit(Pass.NullPass):
    def __init__(self):
        self.productVisit = None
        Pass.NullPass.__init__(self)

    def doProduct(self, product, before):
        if before:
            self.productVisit = product.visit()
        else:
            self.productVisit = None

    def doProductFile(self, file):
        hstFile = HstFilename.HstFilename(file.fullFilepath())
        self.assertEquals(self.productVisit, hstFile.visit(),
                          'Unexpected visit value for file %s' % repr(file))


class BundleContainsOneSingleHstInternalProposalId(Pass.NullPass):
    def __init__(self):
        self.hstInternalProposalIds = None
        self.bundleProposalId = None
        Pass.NullPass.__init__(self)

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


class BundleContainsBundleXml(Pass.LimitedReportingPass):
    def __init__(self):
        Pass.LimitedReportingPass.__init__(self)
        self.sawBundleXml = None

    def doBundle(self, bundle, before):
        if before:
            self.sawBundleXml = False
        else:
            if not self.sawBundleXml:
                self.report('Bundle missing bundle.xml file.')

    def doBundleFile(self, file):
        if file.basename() == 'bundle.xml':
            self.sawBundleXml = True


class CollectionContainsCollectionXml(Pass.LimitedReportingPass):
    def __init__(self):
        Pass.LimitedReportingPass.__init__(self)
        self.sawCollectionXml = None

    def doCollection(self, collection, before):
        if before:
            self.sawCollectionXml = False
            self.collectionInvName = 'collection_%s_inventory.tab' % \
                collection.suffix()
            self.collectionXmlName = 'collection_%s.xml' % collection.suffix()
        else:
            if not self.sawCollectionXml:
                self.report('Collection missing %s file.' %
                            self.collectionXmlName)

    def doCollectionFile(self, file):
        if file.basename() == self.collectionXmlName:
            self.sawCollectionXml = True
        elif file.basename() == self.collectionInventoryName:
            self.sawCollectionInv = True

stdValidation = Pass.CompositePass([
        CountFilesPass(),
        ProductFilesHaveBundleProposalId(),
        ProductFilesHaveCollectionSuffix(),
        ProductFilesHaveProductVisit(),
        BundleContainsOneSingleHstInternalProposalId(),
        BundleContainsBundleXml(),
        CollectionContainsCollectionXml(),
        ])
