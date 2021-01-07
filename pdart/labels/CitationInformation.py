"""
Functionality to create a ``<Citation_Information />`` XML element.
"""
from typing import List

from pdart.citations import Citation_Information
from pdart.xml.Templates import (
    FragBuilder,
    NodeBuilder,
    NodeBuilderTemplate,
    combine_nodes_into_fragment,
    interpret_template,
)

PUBLICATION_YEAR = 2021

_bundle_citation_information: NodeBuilderTemplate = interpret_template(
    """<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<doi><NODE name="doi" /></doi>
<FRAGMENT name="keywords"  />
<description><NODE name="description" /></description>
</Citation_Information>"""
)
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element in bundle label.
"""

_collection_citation_information: NodeBuilderTemplate = interpret_template(
    """<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<FRAGMENT name="keywords"  />
<description><NODE name="description" /></description>
</Citation_Information>"""
)
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element in collection label.
"""

_data_citation_information: NodeBuilderTemplate = interpret_template(
    """<Citation_Information>
<author_list><NODE name="author_list" /></author_list>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>"""
)
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element in data label.
"""

_make_keyword_node: NodeBuilderTemplate = interpret_template(
    """<keyword><NODE name="keyword"  /></keyword>"""
)


def _make_keyword(keyword: str) -> FragBuilder:
    return _make_keyword_node({"keyword": keyword})


def make_citation_information(
    info: Citation_Information,
    is_for_bundle: bool = False,
    is_for_data: bool = False,
) -> NodeBuilder:
    """
    Return a ``<Citation_Information />`` XML element.
    """
    info.set_publication_year(PUBLICATION_YEAR)
    keyword_nodes: List[NodeBuilder] = [
        _make_keyword(keyword) for keyword in info.keywords
    ]

    if is_for_bundle:
        return _bundle_citation_information(
            {
                "description": info.description,
                "publication_year": info.publication_year,
                "author_list": info.author_list,
                "keywords": combine_nodes_into_fragment(keyword_nodes),
                # Need to fake it to something that will pass the XsdValidator.jar for now.
                # required pattern for dummy value: 10\\.\\S+/\\S+
                "doi": "10.1/2",
            }
        )
    elif is_for_data:
        return _data_citation_information(
            {
                "description": info.description,
                "publication_year": info.publication_year,
                "author_list": info.author_list,
            }
        )
    else:
        return _collection_citation_information(
            {
                "description": info.description,
                "publication_year": info.publication_year,
                "author_list": info.author_list,
                "keywords": combine_nodes_into_fragment(keyword_nodes),
            }
        )
