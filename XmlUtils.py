import abc
import xml.dom


class XmlUtils(object):
    """An abstract class providing XML functionality."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, document):
        """Store the document (which provides the XML functionality)."""
        self.document = document

    def add_processing_instruction(self, lhs, rhs):
        """Add a processing instruction to the XML document."""
        d = self.document
        return d.appendChild(d.createProcessingInstruction(lhs, rhs))

    def create_child(self, parent, name):
        """
        Create a child node to the parent with the given tag and
        return it.  This allows chaining.
        """
        d = self.document
        return parent.appendChild(d.createElement(name))

    def create_children(self, parent, names):
        """
        Create a child nodes of the parent with the given tags,
        returning them.
        """
        d = self.document
        return [self.create_child(parent, name) for name in names]

    def set_text(self, parent, txt):
        """Create a child text node of the parent, returning it."""
        if txt is not None:
            d = self.document
            return parent.appendChild(d.createTextNode(txt))

    def print_default_xml(self):
        """Pretty-print the XML document."""
        print self.document.toprettyxml(indent='  ',
                                        newl='\n',
                                        encoding='utf-8')

# was_converted
