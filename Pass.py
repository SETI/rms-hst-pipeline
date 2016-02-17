class NullPass(object):
    def __init__(self):
        pass

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
        print self.__class__.__name__ + ':',
        if tag is not None:
            print msg, tag
        else:
            print msg


class CompositePass(NullPass):
    def __init__(self, passes):
        NullPass.__init__(self)
        self.passes = passes

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


def runArchivePasses(archive, p):
    p.doArchive(archive, True)
    for bundle in archive.bundles():
        p.doBundle(bundle, True)
        for file in bundle.files():
            p.doBundleFile(file)
        for collection in bundle.collections():
            p.doCollection(collection, True)
            for file in collection.files():
                p.doCollectionFile(file)
            for product in collection.products():
                p.doProduct(product, True)
                for file in product.files():
                    p.doProductFile(file)
                p.doProduct(product, False)
            p.doCollection(collection, False)
        p.doBundle(bundle, False)
    p.doArchive(archive, False)
