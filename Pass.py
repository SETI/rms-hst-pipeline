class NullPass(object):
    def __init__(self):
        self.contextHolder = None

    def setContextHolder(self, ctxtHolder):
        self.contextHolder = ctxtHolder

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
        print self.contextHolder.context + ':',
        if tag is not None:
            print msg, tag
        else:
            print msg


class CompositePass(NullPass):
    def __init__(self, passes):
        NullPass.__init__(self)
        self.passes = passes

    def setContextHolder(self, ctxtHolder):
        NullPass.setContextHolder(self, ctxtHolder)
        for p in self.passes:
            p.setContextHolder(ctxtHolder)

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


class _ContextHolder(object):
    def __init__(self):
        self.context = None


def runArchivePasses(archive, p):
    ctxtHolder = _ContextHolder()
    p.setContextHolder(ctxtHolder)

    ctxtHolder.context = archive
    p.doArchive(archive, True)
    for bundle in archive.bundles():
        ctxtHolder.context = bundle
        p.doBundle(bundle, True)
        for file in bundle.files():
            ctxtHolder.context = file
            p.doBundleFile(file)
        for collection in bundle.collections():
            ctxtHolder.context = collection
            p.doCollection(collection, True)
            for file in collection.files():
                ctxtHolder.context = file
                p.doCollectionFile(file)
            for product in collection.products():
                ctxtHolder.context = product
                p.doProduct(product, True)
                for file in product.files():
                    ctxtHolder.context = file
                    p.doProductFile(file)
                ctxtHolder.context = product
                p.doProduct(product, False)
            ctxtHolder.context = collection
            p.doCollection(collection, False)
        ctxtHolder.context = bundle
        p.doBundle(bundle, False)
    ctxtHolder.context = archive
    p.doArchive(archive, False)
