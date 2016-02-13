import shutil
import tempfile
import unittest
# We need to use xml.dom instead of the other XML implementations
# because it allows setting ProcessingInstruction on the document.
import xml.dom


class Label:
    def __init__(self):
        self.impl = xml.dom.getDOMImplementation()
        d = self.document = self.impl.createDocument(None, None, None)
        self.__writePIs()
        self.root = d.appendChild(d.createElement('Product_Observational'))

    __schematronHrefs = \
        ['https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch',
         'https://pds.nasa.gov/pds4/schema/develop/img/PDS4_IMG_1500.sch',
         'https://pds.nasa.gov/pds4/schema/develop/geom/PDS4_GEOM_1500.sch',
         'https://pds.nasa.gov/pds4/schema/' +
         'develop/mission/insight/PDS4_INSIGHT_1000.sch',
         'https://pds.nasa.gov/pds4/schema/' +
         'develop/mission/insight/PDS4_INSIGHT_CAMERAS_RAW_1500.sch']

    def __writePIs(self):
        d = self.document
        for href in Label.__schematronHrefs:
            data = ('href="%s" ' +
                    'schematypens="http://purl.oclc.org/dsdl/schematron"') % \
                    href
            d.appendChild(d.createProcessingInstruction('xml-model', data))

    def write(self, filename, pretty=False):
        if pretty:
            self.writePretty(filename)
        else:
            self.writeCompact(filename)

    def writeCompact(self, filename):
        with open(filename, 'w') as f:
            self.document.writexml(f, encoding='utf-8')

    def writePretty(self, filename):
        with open(filename, 'w') as f:
            self.document.writexml(f, encoding='utf-8',
                                   newl='\n', addindent='  ')

    def toString(self, pretty=False):
        if pretty:
            return self.toPrettyString()
        else:
            return self.toCompactString()

    def toCompactString(self):
        return self.document.toxml('utf-8')

    def toPrettyString(self):
        return self.document.toprettyxml(indent='  ', newl='\n',
                                         encoding='utf-8')

############################################################


class TestLabel(unittest.TestCase):
    def testInit(self):
        minimalLabel = Label()
        minimalXml = '<?xml version="1.0" encoding="utf-8"?>' + \
            '<?xml-model href="https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS' + \
            '_1500.sch" schematypens="http://purl.oclc.org/dsdl/schematr' + \
            'on"?><?xml-model href="https://pds.nasa.gov/pds4/schema/dev' + \
            'elop/img/PDS4_IMG_1500.sch" schematypens="http://purl.oclc.' + \
            'org/dsdl/schematron"?><?xml-model href="https://pds.nasa.go' + \
            'v/pds4/schema/develop/geom/PDS4_GEOM_1500.sch" schematypens' + \
            '="http://purl.oclc.org/dsdl/schematron"?><?xml-model href="' + \
            'https://pds.nasa.gov/pds4/schema/develop/mission/insight/PD' + \
            'S4_INSIGHT_1000.sch" schematypens="http://purl.oclc.org/dsd' + \
            'l/schematron"?><?xml-model href="https://pds.nasa.gov/pds4/' + \
            'schema/develop/mission/insight/PDS4_INSIGHT_CAMERAS_RAW_150' + \
            '0.sch" schematypens="http://purl.oclc.org/dsdl/schematron"?' + \
            '><Product_Observational/>'
        self.assertEqual(minimalXml, minimalLabel.toCompactString())

if __name__ == '__main__':
    unittest.main()
