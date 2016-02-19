import abc
import os
import os.path
import xml.dom

import ArchiveComponent
import Info


class LabelMaker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, component, info):
        assert isinstance(component, ArchiveComponent.ArchiveComponent)
        self.component = component
        assert isinstance(info, Info.Info)
        self.info = info
        self.domImpl = xml.dom.getDOMImplementation()
        self.document = self.domImpl.createDocument(None, None, None)
        self.createDefaultXml()

    @abc.abstractmethod
    def createDefaultXml(self):
        pass

    @abc.abstractmethod
    def defaultXmlName(self):
        pass

    def _addProcessingInstruction(self, lhs, rhs):
        d = self.document
        return d.appendChild(d.createProcessingInstruction(lhs, rhs))

    def _createChild(self, parent, name):
        d = self.document
        return parent.appendChild(d.createElement(name))

    def _createChildren(self, parent, names):
        d = self.document
        return [self._createChild(parent, name) for name in names]

    def _setText(self, parent, txt):
        d = self.document
        return parent.appendChild(d.createTextNode(txt))

    def printDefaultXml(self):
        print self.document.toprettyxml(indent='  ',
                                        newl='\n',
                                        encoding='utf-8')

    def createDefaultXmlFile(self, xmlFilepath=None):
        if xmlFilepath is None:
            xmlName = self.defaultXmlName()
            xmlFilepath = os.path.join(self.component.directoryFilepath(),
                                       xmlName)
        with open(xmlFilepath, 'w') as outputFile:
            pretty = True
            if pretty:
                outputFile.write(self.document.toprettyxml(indent='  ',
                                                           newl='\n',
                                                           encoding='utf-8'))
            else:
                outputFile.write(self.document.toxml(encoding='utf-8'))


def xmlSchemaCheck(filepath):
    exitCode = os.system('xmllint --noout --schema "%s" %s' %
                         ('./xml/PDS4_PDS_1500.xsd.xml', filepath))
    return exitCode == 0
