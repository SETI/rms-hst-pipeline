from pdart.xml.Templates import interpret_template


placeholder_citation_information = interpret_template(
    """<Citation_Information>
<publication_year>2000</publication_year>
<description>### placeholder for \
citation_information_description ###</description>
</Citation_Information>""")({})
"""
An interpreted fragment template to create a ``<Citation_Information
/>`` XML element.
"""
# type: NodeBuilder
