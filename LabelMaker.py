import abc
import os
import os.path
import xml.dom

import ArchiveComponent
import Info
import XmlUtils


class LabelMaker(XmlUtils.XmlUtils):
    def __init__(self, component, info):
        assert isinstance(component, ArchiveComponent.ArchiveComponent)
        self.component = component
        assert isinstance(info, Info.Info)
        self.info = info
        document = xml.dom.getDOMImplementation().createDocument(None,
                                                                 None,
                                                                 None)
        XmlUtils.XmlUtils.__init__(self, document)

        self.createDefaultXml()

    @abc.abstractmethod
    def createDefaultXml(self):
        pass

    @abc.abstractmethod
    def defaultXmlName(self):
        pass


def xmlSchemaCheck(filepath):
    cmdTemplate = 'xmllint --noout --schema %s %s'
    exitCode = os.system(cmdTemplate %
                         ('./xml/PDS4_PDS_1500.xsd.xml', filepath))
    return exitCode == 0


def schematronCheck(filepath):
    cmdTemplate = 'java -jar probatron.jar %s %s' + \
        ' | xmllint -format -' + \
        ' | python Schematron.py'
    exitCode = os.system(cmdTemplate %
                         (filepath, './xml/PDS4_PDS_1500.sch.xml'))
    return exitCode == 0
