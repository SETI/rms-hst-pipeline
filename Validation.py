class NullValidation(object):
    def __init__(self):
        pass

    def doArchive(self, archive, before):
        pass

    def doBundle(self, bundle, before):
        pass

    def doCollection(self, collection, before):
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


class CompositeValidation(NullValidation):
    def __init__(self, validations):
        NullValidation.__init__(self)
        self.validations = validations

    def doArchive(self, archive, before):
        if before:
            for v in self.validations:
                v.doArchive(archive, before)
        else:
            for v in reversed(self.validations):
                v.doArchive(archive, before)

    def doBundle(self, bundle, before):
        if before:
            for v in self.validations:
                v.doBundle(bundle, before)
        else:
            for v in reversed(self.validations):
                v.doBundle(bundle, before)

    def doCollection(self, collection, before):
        if before:
            for v in self.validations:
                v.doCollection(collection, before)
        else:
            for v in reversed(self.validations):
                v.doCollection(collection, before)

    def doProduct(self, product, before):
        if before:
            for v in self.validations:
                v.doProduct(product, before)
        else:
            for v in reversed(self.validations):
                v.doProduct(product, before)

    def doProductFile(self, file):
        for v in self.validations:
            v.doProductFile(file)


def runArchiveValidation(archive, v):
    v.doArchive(archive, True)
    for bundle in archive.bundles():
        v.doBundle(bundle, True)
        for collection in bundle.collections():
            v.doCollection(collection, True)
            for product in collection.products():
                v.doProduct(product, True)
                for file in product.files():
                    v.doProductFile(file)
                v.doProduct(product, False)
            v.doCollection(collection, False)
        v.doBundle(bundle, False)
    v.doArchive(archive, False)
