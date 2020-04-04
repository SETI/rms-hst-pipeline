"""
Functionality to create a ``<Citation_Information />`` XML element.
"""

from pdart.citations import Citation_Information
from pdart.xml.Templates import interpret_template

from pdart.xml.Templates import NodeBuilder, NodeBuilderTemplate

_citation_information: NodeBuilderTemplate = interpret_template(
    """<Citation_Information>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>"""
)
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element.
"""


def make_citation_information(info: Citation_Information) -> NodeBuilder:
    """
    Return a placeholder ``<Citation_Information />`` XML element.
    """
    return _citation_information(
        {"description": info.description, "publication_year": info.publication_year}
    )
