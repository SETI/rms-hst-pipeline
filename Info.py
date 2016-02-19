class Info(object):
    def pdsNamespaceUrl(self):
        return 'http://pds.nasa.gov/pds4/pds/v1'

    def informationModelVersion(self):
        return '1.5.0.0'

    def versionID(self):
        return '1.0'

    def PLACEHOLDER(self, tag):
        return '### placeholder for %s.%s() ###' % \
            (self.__class__.__name__, tag)
