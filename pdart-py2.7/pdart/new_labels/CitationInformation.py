"""
Functionality to create a ``<Citation_Information />`` XML element.
"""
from typing import TYPE_CHECKING

from Citation_Information import Citation_Information
from pdart.new_labels.Placeholders import known_placeholder, placeholder_year
from pdart.xml.Templates import interpret_template

if TYPE_CHECKING:
    from pdart.xml.Templates import NodeBuilder, NodeBuilderTemplate

_citation_information = interpret_template(
    """<Citation_Information>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>""")  # type: NodeBuilderTemplate
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element.
"""


def make_citation_information(info):
    # type: (Citation_Information) -> NodeBuilder
    """
    Return a placeholder ``<Citation_Information />`` XML element.
    """
    return _citation_information({
        'description': info.description,
        'publication_year': info.publication_year
    })
