import abc
import xml.dom


class XmlUtils(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, document):
        self.document = document

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
        if txt is not None:
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
