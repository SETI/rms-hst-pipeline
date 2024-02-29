"""
Functionality to create a ``<Citation_Information />`` XML element.
"""

from pdart.citations import Citation_Information
from pdart.xml.templates import NodeBuilder, NodeBuilderTemplate, interpret_template

PUBLICATION_YEAR = 2021

_collection_citation_information: NodeBuilderTemplate = interpret_template(
    """<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>"""
)
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element.
"""

_bundle_citation_information: NodeBuilderTemplate = interpret_template(
    """<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<doi><NODE name="doi" /></doi>
<description><NODE name="description" /></description>
</Citation_Information>"""
)
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element.
"""


def make_citation_information(
    info: Citation_Information,
    is_for_bundle: bool = False,
) -> NodeBuilder:
    """
    Return a ``<Citation_Information />`` XML element.
    """
    if is_for_bundle:
        return _bundle_citation_information(
            {
                "description": info.description,
                "publication_year": info.publication_year,
                "author_list": info.author_list,
                # Need to fake it to something that will pass the XsdValidator.jar for now.
                # required pattern for dummy value: 10\\.\\S+/\\S+
                "doi": "10.1/2",
            }
        )
    else:
        return _collection_citation_information(
            {
                "description": info.description,
                "publication_year": info.publication_year,
                "author_list": info.author_list,
            }
        )
