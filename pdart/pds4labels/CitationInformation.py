from pdart.pds4labels.Placeholders import *
from pdart.xml.Templates import *


_placeholder_citation_information = interpret_template(
    """<Citation_Information>
<publication_year><NODE name="publication_year" /></publication_year>
<description><NODE name="description" /></description>
</Citation_Information>""")
"""
An interpreted template to create a ``<Citation_Information
/>`` XML element.
"""
# type: NodeBuilderTemplate


def make_placeholder_citation_information(component_id):
    # type: (unicode) -> NodeBuilder
    pub_year = get_citation_information_publication_year(component_id)
    return _placeholder_citation_information({
            'description': get_citation_information_description(component_id),
            'publication_year': pub_year
            })


def get_citation_information_description(component_id):
    # type: (unicode) -> unicode
    return known_placeholder(component_id, 'Citation_Information/description')


def get_citation_information_publication_year(component_id):
    # type: (unicode) -> unicode
    return placeholder_year(component_id,
                            'Citation_Information/publication_year')
