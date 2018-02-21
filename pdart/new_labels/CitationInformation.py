"""
Functionality to create a ``<Citation_Information />`` XML element.
"""
from pdart.new_labels.Placeholders import *
from pdart.xml.Templates import *

_placeholder_citation_information = interpret_template(
    """<Citation_Information>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>""")
# type: NodeBuilderTemplate
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element.
"""


def make_placeholder_citation_information(component_id):
    # type: (unicode) -> NodeBuilder
    """
    Return a placeholder ``<Citation_Information />`` XML element.
    """
    pub_year = _get_citation_information_publication_year(component_id)
    return _placeholder_citation_information({
        'description': _get_citation_information_description(component_id),
        'publication_year': pub_year
    })


def _get_citation_information_description(component_id):
    # type: (unicode) -> unicode
    """
    Return a placeholder for the description text of a
    ``<Citation_Information />`` XML element.
    """
    return known_placeholder(component_id, 'Citation_Information/description')


def _get_citation_information_publication_year(component_id):
    # type: (unicode) -> unicode
    """
    Return a placeholder for the publication year of a
    ``<Citation_Information />`` XML element.
    """
    return placeholder_year(component_id,
                            'Citation_Information/publication_year')
