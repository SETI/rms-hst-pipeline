import abc

import XmlUtils


class XmlMaker(XmlUtils.XmlUtils):
    """
    An abstract class that can create portions of XML documents.

    In the initializer, you add any extra information you need to
    build the XML element(s).

    In create_xml(), you create the needed XML.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, document):
        assert document
        super(XmlMaker, self).__init__(document)

    @abc.abstractmethod
    def create_xml(self, parent):
        pass
