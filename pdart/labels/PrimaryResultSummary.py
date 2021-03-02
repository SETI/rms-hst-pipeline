"""
Templates to create an ``<Primary_Result_Summary />`` XML element.
Roll-up: One <Primary_Result_Summary /> for bundle, data colleciton, and data.
There can be multiple processing levels, wavelength names, and the title are
different in bundle, data collection, and data level.
"""
from typing import Any, List, Dict

from pdart.xml.Templates import (
    combine_nodes_into_fragment,
    FragBuilder,
    NodeBuilder,
    NodeBuilderTemplate,
    interpret_template,
)

_primary_result_summary: NodeBuilderTemplate = interpret_template(
    """<Primary_Result_Summary>
      <purpose>Science</purpose>
      <processing_level><NODE name="processing_level"/></processing_level>
      <description><NODE name="description"/></description>
      <Science_Facets>
        <FRAGMENT name="wavelength_range"/>
        <discipline_name>Ring-Moon Systems</discipline_name>
      </Science_Facets>
    </Primary_Result_Summary>"""
)
"""
An interpreted fragment template to create an ``<Primary_Result_Summary />``
XML element.
"""

_make_wavelength_range_node: NodeBuilderTemplate = interpret_template(
    """<wavelength_range><NODE name="wavelength_range"/></wavelength_range>"""
)


def _make_wavelength_range(wavelength_range: str) -> FragBuilder:
    return _make_wavelength_range_node({"wavelength_range": wavelength_range})


def primary_result_summary(result_dict: Dict[str, Any]) -> NodeBuilder:
    """
    Given an instrument, return an interpreted fragment template to
    create an ``<Primary_Result_Summary />`` XML element.
    """
    wavelength_range_list = result_dict["wavelength_range"]
    wavelength_range_nodes: List[NodeBuilder] = [
        _make_wavelength_range(wavelength_range)
        for wavelength_range in wavelength_range_list
    ]

    return _primary_result_summary(
        {
            "processing_level": result_dict["processing_level"],
            "description": result_dict["description"],
            "wavelength_range": combine_nodes_into_fragment(wavelength_range_nodes),
        }
    )
