"""Templates to create an ``<Investigation_Area />`` XML element."""
from typing import Any, Dict

from pdart.xml.Templates import NodeBuilder, NodeBuilderTemplate, interpret_template


_investigation_area: NodeBuilderTemplate = interpret_template(
    """<Investigation_Area>
      <name><NODE name="Investigation_Area_name" /></name>
      <type>Individual Investigation</type>
      <Internal_Reference>
        <lidvid_reference><NODE name="investigation_lidvid" />\
</lidvid_reference>
        <reference_type><NODE name="reference_type" />_to_investigation\
</reference_type>
      </Internal_Reference>
    </Investigation_Area>"""
)
"""
An interpreted fragment template to create an ``<Investigation_Area />``
XML element.
"""


def investigation_area(
    Investigation_Area_name: str,
    investigation_lidvid: str,
    reference_type: str,
) -> NodeBuilder:
    """
    Given an instrument, return an interpreted fragment template to
    create an ``<Investigation_Area />`` XML element.
    """
    return _investigation_area(
        {
            "Investigation_Area_name": Investigation_Area_name,
            "investigation_lidvid": investigation_lidvid,
            "reference_type": reference_type,
        }
    )
