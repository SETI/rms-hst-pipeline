"""Templates to create an ``<Primary_Result_Summary />`` XML element."""
from typing import Any, Dict

from pdart.xml.Templates import NodeBuilder, NodeBuilderTemplate, interpret_template

_primary_result_summary: NodeBuilderTemplate = interpret_template(
    """<Primary_Result_Summary>
      <purpose>Science</purpose>
      <processing_level><NODE name="processing_level"/></processing_level>
      <description><NODE name="description"/></description>
      <Science_Facets>
        <wavelength_range><NODE name="wavelength_range"/></wavelength_range>
        <discipline_name>Ring-Moon Systems</discipline_name>
      </Science_Facets>
    </Primary_Result_Summary>"""
)
"""
An interpreted fragment template to create an ``<Primary_Result_Summary />``
XML element.
"""


def primary_result_summary(result_dict: Dict[str, Any]) -> NodeBuilder:
    """
    Given an instrument, return an interpreted fragment template to
    create an ``<Primary_Result_Summary />`` XML element.
    """
    return _primary_result_summary(
        {
            "processing_level": result_dict["processing_level"],
            "description": result_dict["description"],
            "wavelength_range": result_dict["wavelength_range"],
        }
    )
