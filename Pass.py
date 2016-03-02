import abc

import Reporter


class Pass(object):
    """
    A set of actions to be performed on a FileArchive.  This is an
    abstract class.
    """

    def __init__(self):
        """Create a Pass object."""
        self.pass_runner = None

    def set_pass_runner(self, pass_runner):
        """
        Set the PassRunner.  It provides access to the Reporter and
        the context (location in the archive) while running the pass.
        """
        self.pass_runner = pass_runner

    @abc.abstractmethod
    def do_archive(self, archive, before):
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
    def do_bundle(self, bundle, before):
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
    def do_bundle_file(self, file):
        """Perform actions for a file in the Bundle directory."""
        pass

    @abc.abstractmethod
    def do_collection(self, collection, before):
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
    def do_collection_file(self, file):
        """Perform actions for a file in the Collection directory."""
        pass

    @abc.abstractmethod
    def do_product(self, product, before):
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
    def do_product_file(self, file):
        """Perform actions for a file in the Product directory."""
        pass

    def assert_equals(self, expected, actual, tag=None):
        """
        Do nothing if the two objects are equal; if not, report that
        through the Reporter.  An optional string tag may be provided;
        it will be added to the report.
        """
        if expected != actual:
            msg = 'expected %r; got %r.' % (expected, actual)
            self.report(msg, tag)

    def assert_boolean(self, bool_value, tag=None):
        """
        Do nothing if the bool_value is true; if not, report that
        through the Rporter.  An optional string tag may be provided;
        it will be added to the report.
        """
        if not bool_value:
            msg = 'assertion failed'
            self.report(msg, tag)

    def report(self, msg, tag=None):
        """Report through the reporter."""
        self.pass_runner.report(self.__class__.__name__, msg, tag)


class NullPass(Pass):
    """A Pass that does nothing.  Useful as a base class."""

    def __init__(self):
        super(NullPass, self).__init__()

    def do_archive(self, archive, before):
        pass

    def do_bundle(self, bundle, before):
        pass

    def do_bundle_file(self, file):
        pass

    def do_collection(self, collection, before):
        pass

    def do_collection_file(self, file):
        pass

    def do_product(self, product, before):
        pass

    def do_product_file(self, file):
        pass


class LimitedReportingPass(NullPass):
    """
    A Pass that stops reporting after a certain number of reports.
    Useful as a base class, to keep from flooding output with needless
    messages.  "Shut up; I heard you the first twenty times."
    """

    def __init__(self, report_limit=8):
        """Create a pass with a given reporting limit.  Default is 8."""
        super(LimitedReportingPass, self).__init__()
        self.report_count = 0
        self.report_limit = report_limit

    def past_limit(self):
        return self.report_count >= self.report_limit

    def report(self, msg, tag=None):
        self.report_count += 1
        if self.report_count < self.report_limit:
            NullPass.report(self, msg, tag)
        elif self.report_count == self.report_limit:
            NullPass.report(self, msg, tag)
            NullPass.report(self,
                            'Reached reporting limit (=%s).' %
                            self.report_limit)
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

    def set_pass_runner(self, pass_runner):
        NullPass.set_pass_runner(self, pass_runner)
        for p in self.passes:
            p.set_pass_runner(pass_runner)

    def do_archive(self, archive, before):
        if before:
            for p in self.passes:
                p.do_archive(archive, before)
        else:
            for p in reversed(self.passes):
                p.do_archive(archive, before)

    def do_bundle(self, bundle, before):
        if before:
            for p in self.passes:
                p.do_bundle(bundle, before)
        else:
            for p in reversed(self.passes):
                p.do_bundle(bundle, before)

    def do_bundle_file(self, file):
        for p in self.passes:
            p.do_bundle_file(file)

    def do_collection(self, collection, before):
        if before:
            for p in self.passes:
                p.do_collection(collection, before)
        else:
            for p in reversed(self.passes):
                p.do_collection(collection, before)

    def do_collection_file(self, file):
        for p in self.passes:
            p.do_collection_file(file)

    def do_product(self, product, before):
        if before:
            for p in self.passes:
                p.do_product(product, before)
        else:
            for p in reversed(self.passes):
                p.do_product(product, before)

    def do_product_file(self, file):
        for p in self.passes:
            p.do_product_file(file)


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
        p.set_pass_runner(self)

        self.reporter.begin_reporting()
        self.context = archive
        p.do_archive(archive, True)
        for bundle in archive.bundles():
            self.context = bundle
            p.do_bundle(bundle, True)
            for file in bundle.files():
                self.context = file
                p.do_bundle_file(file)
            for collection in bundle.collections():
                self.context = collection
                p.do_collection(collection, True)
                for file in collection.files():
                    self.context = file
                    p.do_collection_file(file)
                for product in collection.products():
                    self.context = product
                    p.do_product(product, True)
                    for file in product.files():
                        self.context = file
                        p.do_product_file(file)
                    self.context = product
                    p.do_product(product, False)
                self.context = collection
                p.do_collection(collection, False)
            self.context = bundle
            p.do_bundle(bundle, False)
        self.context = archive
        p.do_archive(archive, False)
        self.reporter.end_reporting()
