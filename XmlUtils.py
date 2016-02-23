import abc
import xml.dom


class XmlUtils(object):
    """An abstract class providing XML functionality."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, document):
        """Store the document (which provides the XML functionality)."""
        self.document = document

    def addProcessingInstruction(self, lhs, rhs):
        """Add a processing instruction to the XML document."""
        d = self.document
        return d.appendChild(d.createProcessingInstruction(lhs, rhs))

    def createChild(self, parent, name):
        """
        Create a child node to the parent with the given tag and
        return it.  This allows chaining.
        """
        d = self.document
        return parent.appendChild(d.createElement(name))

    def createChildren(self, parent, names):
        """
        Create a child nodes of the parent with the given tags,
        returning them.
        """
        d = self.document
        return [self.createChild(parent, name) for name in names]

    def setText(self, parent, txt):
        """Create a child text node of the parent, returning it."""
        if txt is not None:
            d = self.document
            return parent.appendChild(d.createTextNode(txt))

    def printDefaultXml(self):
        """Pretty-print the XML document."""
        print self.document.toprettyxml(indent='  ',
                                        newl='\n',
                                        encoding='utf-8')

    def createDefaultXmlFile(self, xmlFilepath=None):
        """
        Pretty-print the XML document to a file.  If no filepath is
        given, it will be written to the default label filepath for
        the component.
        """
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
