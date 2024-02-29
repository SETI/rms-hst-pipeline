from typing import Optional

from xml.dom.minidom import Document, Node

class DocumentType(object): ...

class DomImplementation(object):
    def createDocument(
        self,
        namespaceUri: Optional[str],
        qualifiedName: Optional[str],
        doctype: Optional[DocumentType],
    ) -> Document: ...

def getDOMImplementation() -> DomImplementation: ...

__all__ = ["Document", "Node"]
