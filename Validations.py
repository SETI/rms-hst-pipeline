import os.path
import re
import sys

import pyfits

import HstFilename
import Pass


class CountFilesPass(Pass.NullPass):
    def __init__(self):
        self.file_count = None
        super(CountFilesPass, self).__init__()

    def do_archive(self, archive, before):
        if before:
            self.file_count = 0
        else:
            print >> sys.stderr, 'Saw %d files.' % self.file_count
            self.file_count = None

    def do_product_file(self, file):
        self.file_count += 1
        if self.file_count % 200 == 0:
            print >> sys.stderr, 'Saw %d files.' % self.file_count


class ProductFilesHaveCollectionSuffix(Pass.NullPass):
    def __init__(self):
        self.collection_suffix = None
        super(ProductFilesHaveCollectionSuffix, self).__init__()

    def do_collection(self, collection, before):
        if before:
            self.collection_suffix = collection.suffix()
        else:
            self.collection_suffix = None

    def do_product_file(self, file):
        # get file suffix
        file_suffix = HstFilename.HstFilename(file.basename).suffix()
        self.assert_equals(self.collection_suffix, file_suffix,
                           'Unexpected suffix for file %r' % file)


class ProductFilesHaveBundleProposalId(Pass.NullPass):
    def __init__(self):
        self.bundle_proposal_id = None
        super(ProductFilesHaveBundleProposalId, self).__init__()

    def do_bundle(self, bundle, before):
        if before:
            self.bundle_proposal_id = bundle.proposal_id()
        else:
            self.bundle_proposal_id = None

    def do_product_file(self, file):
        try:
            proposid = pyfits.getval(file.full_filepath(), 'PROPOSID')
        except IOError as e:
            # We know that much (all?) of the contents of hst_00000
            # are there due to IOErrors, so let's not report them.
            # Report others, though.
            if self.bundle_proposal_id != 0:
                self.report('IOError %s reading file %s of product %s' %
                            (e, file, file.component))
            proposid = None
        except KeyError:
            proposid = None

        # if it exists, ensure it matches the bundle_proposal_id
        if proposid is not None:
            self.assert_equals(self.bundle_proposal_id, proposid)


class ProductFilesHaveProductVisit(Pass.NullPass):
    def __init__(self):
        self.product_visit = None
        super(ProductFilesHaveProductVisit, self).__init__()

    def do_product(self, product, before):
        if before:
            self.product_visit = product.visit()
        else:
            self.product_visit = None

    def do_product_file(self, file):
        hstFile = HstFilename.HstFilename(file.full_filepath())
        self.assert_equals(self.product_visit, hstFile.visit(),
                           'Unexpected visit value for file %r' % file)


class BundleContainsOneSingleHstInternalProposalId(Pass.NullPass):
    def __init__(self):
        self.hst_internal_proposal_ids = None
        self.bundle_proposal_id = None
        super(BundleContainsOneSingleHstInternalProposalId, self).__init__()

    def do_product_file(self, file):
        hst_file = HstFilename.HstFilename(file.full_filepath())
        self.hst_internal_proposal_ids.add(hst_file.hst_internal_proposal_id())

    def do_bundle(self, bundle, before):
        if before:
            self.bundle_proposal_id = bundle.proposal_id()
            self.hst_internal_proposal_ids = set()
        else:
            # TODO It seems that hst_00000 is a grab bag of
            # lost files.  This needs to be fixed.
            # Otherwise...
            if self.bundle_proposal_id != 0:
                # Assert that for any bundle, all of its
                # files belong to the same project, using the
                # HST internal proposal ID codes.
                self.assert_equals(1, len(self.hst_internal_proposal_ids),
                                   'In bundle %s, saw HST proposal ids %s.' %
                                   (bundle,
                                    list(self.hst_internal_proposal_ids)))

            self.hst_internal_proposal_ids = None
            self.bundle_proposal_id = None


class BundleContainsBundleXml(Pass.LimitedReportingPass):
    def __init__(self):
        self.saw_bundle_xml = None
        super(BundleContainsBundleXml, self).__init__()

    def do_bundle(self, bundle, before):
        if before:
            self.saw_bundle_xml = False
        else:
            if not self.saw_bundle_xml:
                self.report('Bundle missing bundle.xml file.')

    def do_bundle_file(self, file):
        if file.basename() == 'bundle.xml':
            self.saw_bundle_xml = True


class CollectionContainsCollectionXml(Pass.LimitedReportingPass):
    def __init__(self):
        self.saw_collection_xml = None
        super(CollectionContainsCollectionXml, self).__init__()

    def do_collection(self, collection, before):
        if before:
            self.saw_collection_xml = False
            self.collection_inv_name = 'collection_%s_inventory.tab' % \
                collection.suffix()
            self.collection_xml_name = 'collection_%s.xml' % \
                collection.suffix()
        else:
            if not self.saw_collection_xml:
                self.report('Collection missing %s file.' %
                            self.collection_xml_name)

    def do_collection_file(self, file):
        if file.basename() == self.collection_xml_name:
            self.saw_collection_xml = True
        elif file.basename() == self.collection_inv_name:
            self.saw_collection_inv = True

std_validation = Pass.CompositePass([
        CountFilesPass(),
        ProductFilesHaveBundleProposalId(),
        ProductFilesHaveCollectionSuffix(),
        ProductFilesHaveProductVisit(),
        BundleContainsOneSingleHstInternalProposalId(),
        # BundleContainsBundleXml(),
        # CollectionContainsCollectionXml(),
        ])
