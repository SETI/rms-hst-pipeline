import abc

import Reporter


class NullPass(object):
    def __init__(self):
        self.passRunner = None

    def setPassRunner(self, passRunner):
        self.passRunner = passRunner

    def doArchive(self, archive, before):
        pass

    def doBundle(self, bundle, before):
        pass

    def doBundleFile(self, file):
        pass

    def doCollection(self, collection, before):
        pass

    def doCollectionFile(self, file):
        pass

    def doProduct(self, product, before):
        pass

    def doProductFile(self, file):
        pass

    def assertEquals(self, expected, actual, tag=None):
        if expected != actual:
            msg = 'expected %s; got %s.' % (repr(expected), repr(actual))
            self.report(msg, tag)

    def report(self, msg, tag=None):
        self.passRunner.report(self.__class__.__name__, msg, tag)


class LimitedReportingPass(NullPass):
    def __init__(self, reportLimit=8):
        NullPass.__init__(self)
        self.reportCount = 0
        self.reportLimit = reportLimit

    def report(self, msg, tag=None):
        self.reportCount += 1
        if self.reportCount < self.reportLimit:
            NullPass.report(self, msg, tag)
        elif self.reportCount == self.reportLimit:
            NullPass.report(self, msg, tag)
            NullPass.report(self,
                            'Reached reporting limit (=%s).' %
                            self.reportLimit)
        else:
            pass


class CompositePass(NullPass):
    def __init__(self, passes):
        NullPass.__init__(self)
        self.passes = passes

    def setPassRunner(self, passRunner):
        NullPass.setPassRunner(self, passRunner)
        for p in self.passes:
            p.setPassRunner(passRunner)

    def doArchive(self, archive, before):
        if before:
            for p in self.passes:
                p.doArchive(archive, before)
        else:
            for p in reversed(self.passes):
                p.doArchive(archive, before)

    def doBundle(self, bundle, before):
        if before:
            for p in self.passes:
                p.doBundle(bundle, before)
        else:
            for p in reversed(self.passes):
                p.doBundle(bundle, before)

    def doBundleFile(self, file):
        for p in self.passes:
            p.doBundleFile(file)

    def doCollection(self, collection, before):
        if before:
            for p in self.passes:
                p.doCollection(collection, before)
        else:
            for p in reversed(self.passes):
                p.doCollection(collection, before)

    def doCollectionFile(self, file):
        for p in self.passes:
            p.doCollectionFile(file)

    def doProduct(self, product, before):
        if before:
            for p in self.passes:
                p.doProduct(product, before)
        else:
            for p in reversed(self.passes):
                p.doProduct(product, before)

    def doProductFile(self, file):
        for p in self.passes:
            p.doProductFile(file)


class PassRunner(object):
    def __init__(self, reporter=None):
        self.context = None
        if reporter is None:
            reporter = Reporter.StdoutReporter()
        assert isinstance(reporter, Reporter.Reporter)
        self.reporter = reporter

    def report(self, pass_, msg, tag):
        self.reporter.report(pass_, repr(self.context), msg, tag)

    def run(self, archive, p):
        p.setPassRunner(self)

        self.context = archive
        p.doArchive(archive, True)
        for bundle in archive.bundles():
            self.context = bundle
            p.doBundle(bundle, True)
            for file in bundle.files():
                self.context = file
                p.doBundleFile(file)
            for collection in bundle.collections():
                self.context = collection
                p.doCollection(collection, True)
                for file in collection.files():
                    self.context = file
                    p.doCollectionFile(file)
                for product in collection.products():
                    self.context = product
                    p.doProduct(product, True)
                    for file in product.files():
                        self.context = file
                        p.doProductFile(file)
                    self.context = product
                    p.doProduct(product, False)
                self.context = collection
                p.doCollection(collection, False)
            self.context = bundle
            p.doBundle(bundle, False)
        self.context = archive
        p.doArchive(archive, False)
