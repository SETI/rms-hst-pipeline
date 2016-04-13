import pprint  # TODO remove
import sys
import traceback

import pyfits

import BundleLabelMaker
import CollectionLabelMaker
import Pass
import ProductLabelMaker
import ProductPass
import pdart.pds4.HstFilename
import pdart.pds4.LID
import pdart.xml.Schema


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
        file_suffix = \
            pdart.pds4.HstFilename.HstFilename(file.basename).suffix()
        self.assert_equals(self.collection_suffix, file_suffix,
                           'Unexpected suffix for file %r' % file)


class ProductFilesHaveBundleProposalId(Pass.NullPass):
    def __init__(self):
        self.bundle_proposal_id = None
        self.collection_suffix = None
        super(ProductFilesHaveBundleProposalId, self).__init__()

    def do_bundle(self, bundle, before):
        if before:
            self.bundle_proposal_id = bundle.proposal_id()
        else:
            self.bundle_proposal_id = None

    def do_collection(self, collection, before):
        if before:
            self.collection_suffix = collection.suffix()
        else:
            self.collection_suffix = None

    def do_product_file(self, file):
        try:
            proposid = pyfits.getval(file.full_filepath(), 'PROPOSID')
        except IOError:
            # We ignore IOErrors as they'll get picked up when we try
            # to build labels.
            proposid = None
        except KeyError:
            proposid = None

        # if it exists, ensure it matches the bundle_proposal_id
        if proposid is not None:
            if self.collection_suffix == 'lrc':
                self.assert_equals(0, proposid)
            else:
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
        hstFile = pdart.pds4.HstFilename.HstFilename(file.full_filepath())
        self.assert_equals(self.product_visit, hstFile.visit(),
                           'Unexpected visit value for file %r' % file)


class BundleContainsOneSingleHstInternalProposalId(Pass.NullPass):
    def __init__(self):
        self.hst_internal_proposal_ids = None
        self.bundle_proposal_id = None
        super(BundleContainsOneSingleHstInternalProposalId, self).__init__()

    def do_product_file(self, file):
        hst_file = pdart.pds4.HstFilename.HstFilename(file.full_filepath())
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
        self.collection_inv_name = None
        self.collection_xml_name = None
        self.saw_collection_inv = None
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


class CorrectLabel(Pass.LimitedReportingPass):
    def __init__(self):
        super(CorrectLabel, self).__init__()

    def check_label(self, lid, filename):
        assert isinstance(lid, pdart.pds4.LID.LID)
        if not self.past_limit():
            failures = pdart.xml.Schema.xml_schema_failures(filename)
            if failures:
                self.report('Label for %s failed XML schema test' % lid,
                            failures)
            else:
                failures = pdart.xml.Schema.schematron_failures(filename)
                if failures:
                    self.report('Label for %s failed Schematron test' % lid,
                                failures)


class CorrectBundleLabel(CorrectLabel):
    def __init__(self):
        super(CorrectBundleLabel, self).__init__()

    def do_bundle(self, bundle, before):
        if before and not self.past_limit():
            lm = BundleLabelMaker.BundleLabelMaker(bundle)
            filename = '/tmp/bundle.xml'
            lm.write_xml_to_file(filename)
            self.check_label(bundle.lid, filename)


class CorrectCollectionLabel(CorrectLabel):
    def __init__(self):
        super(CorrectCollectionLabel, self).__init__()

    def do_collection(self, collection, before):
        if before and not self.past_limit():
            lm = CollectionLabelMaker.CollectionLabelMaker(collection)
            filename = '/tmp/collection.xml'
            lm.write_xml_to_file(filename)
            self.check_label(collection.lid, filename)


class CorrectProductLabel(CorrectLabel):
    def __init__(self):
        super(CorrectProductLabel, self).__init__()

    def do_product(self, product, before):
        if before and not self.past_limit():
            try:
                lm = ProductLabelMaker.ProductLabelMaker(product)
            except IOError:
                lm = None
            self.assert_boolean(lm, 'Building xml for %s' % product)
            if lm:
                filename = '/tmp/product.xml'
                lm.write_xml_to_file(filename)
                self.check_label(product.lid, filename)


class DemoProductPass(Pass.NullPass):
    def do_product(self, product, before):
        if before:
            pp = ProductPass.ProductLabelProductPass()
            ppr = ProductPass.ProductPassRunner()
            print 60 * '-'
            print 8 * '-', product
            try:
                res = ppr.run_product(pp, product)
                print "SUCCESS"
                print pprint.PrettyPrinter(indent=2, width=78).pprint(res)
            except:
                print "FAILURE"
                print traceback.format_exc()


std_validation = Pass.CompositePass([
        CountFilesPass(),
        ProductFilesHaveBundleProposalId(),
        ProductFilesHaveCollectionSuffix(),
        ProductFilesHaveProductVisit(),
        BundleContainsOneSingleHstInternalProposalId(),
        CorrectBundleLabel(),
        CorrectCollectionLabel(),
        CorrectProductLabel(),
        # DemoProductPass(),
        # BundleContainsBundleXml(),
        # CollectionContainsCollectionXml(),
        ])
