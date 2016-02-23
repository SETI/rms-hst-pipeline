import abc

import Reporter


class Pass(object):
    """
    A set of actions to be performed on a FileArchive.  This is an
    abstract class.
    """

    def __init__(self):
        """Create a Pass object."""
        self.passRunner = None

    def setPassRunner(self, passRunner):
        """
        Set the PassRunner.  It provides access to the Reporter and
        the context (location in the archive) while running the pass.
        """
        self.passRunner = passRunner

    @abc.abstractmethod
    def doArchive(self, archive, before):
        """
        Perform actions before or after the contents of a FileArchive
        are processed.  This method is called twice; the Boolean
        argument tells you if you're being called before (True) or
        after (False).

        In general, if you don't need information from the contents,
        prefer to act before the FileArchive's contents are processed.
        """
        pass

    @abc.abstractmethod
    def doBundle(self, bundle, before):
        """
        Perform actions before or after the contents of a Bundle are
        processed.  This method is called twice; the Boolean argument
        tells you if you're being called before (True) or after
        (False).

        In general, if you don't need information from the contents,
        prefer to act before the Bundle's contents are processed.
        """
        pass

    @abc.abstractmethod
    def doBundleFile(self, file):
        """Perform actions for a file in the Bundle directory."""
        pass

    @abc.abstractmethod
    def doCollection(self, collection, before):
        """
        Perform actions before or after the contents of a Collection
        are processed.  This method is called twice; the Boolean
        argument tells you if you're being called before (True) or
        after (False).

        In general, if you don't need information from the contents,
        prefer to act before the Collection's contents are processed.
        """
        pass

    @abc.abstractmethod
    def doCollectionFile(self, file):
        """Perform actions for a file in the Collection directory."""
        pass

    @abc.abstractmethod
    def doProduct(self, product, before):
        """
        Perform actions before or after the contents of a Product are
        processed.  This method is called twice; the Boolean argument
        tells you if you're being called before (True) or after
        (False).

        In general, if you don't need information from the contents,
        prefer to act before the Product's contents are processed.
        """
        pass

    @abc.abstractmethod
    def doProductFile(self, file):
        """Perform actions for a file in the Product directory."""
        pass

    def assertEquals(self, expected, actual, tag=None):
        """
        Do nothing if the two objects are equal; if not, report that
        through the Reporter.  An optional string tag may be provided;
        it will be added to the report.
        """
        if expected != actual:
            msg = 'expected %s; got %s.' % (repr(expected), repr(actual))
            self.report(msg, tag)

    def report(self, msg, tag=None):
        """Report through the reporter."""
        self.passRunner.report(self.__class__.__name__, msg, tag)


class NullPass(Pass):
    """A Pass that does nothing.  Useful as a base class."""

    def __init__(self):
        super(NullPass, self).__init__()

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


class LimitedReportingPass(NullPass):
    """
    A Pass that stops reporting after a certain number of reports.
    Useful as a base class, to keep from flooding output with needless
    messages.  "Shut up; I heard you the first twenty times."
    """

    def __init__(self, reportLimit=8):
        """Create a pass with a given reporting limit.  Default is 8."""
        super(LimitedReportingPass, self).__init__()
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
    """
    A Pass composed of a list of passes.  The component passes'
    actions for each level of the archive are run in order for
    "before" calls, and in reverse order for "after" calls.  This
    allows multiple sets of actions to be performed while only making
    one actual pass over the archive.
    """

    def __init__(self, passes):
        """Create a CompositePass with the given list of passes."""
        self.passes = passes
        super(CompositePass, self).__init__()

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
    """
    A object to run a (possible composite) Pass over a FileArchive,
    reporting results through a configurable Reporter.  It provides
    access to the context (location in the archive) while running the
    pass.
    """

    def __init__(self, reporter=None):
        """
        Create a PassRunner.  If no reporter is given, it will report
        using the StdoutReporter.
        """
        self.context = None
        if reporter is None:
            reporter = Reporter.StdoutReporter()
        assert isinstance(reporter, Reporter.Reporter)
        self.reporter = reporter

    def report(self, pass_, msg, tag):
        """Report to the configured Reporter."""
        # TODO I should ensure that these are all strings (or None).
        self.reporter.report(pass_, repr(self.context), msg, tag)

    def run(self, archive, p):
        """Run the Pass on the FileArchive."""
        p.setPassRunner(self)

        self.reporter.beginReporting()
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
        self.reporter.endReporting()
