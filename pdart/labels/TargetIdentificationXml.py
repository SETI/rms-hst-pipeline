"""
Templates to create a ``<Target_Identification />`` XML node for
product labels.
"""

from typing import Dict, List

from pdart.xml.Templates import (
    combine_nodes_into_fragment,
    FragBuilder,
    interpret_template,
    NodeBuilder,
    NodeBuilderTemplate,
)

_make_alternate_designation_node: NodeBuilderTemplate = interpret_template(
    """<alternate_designation><NODE name="alternate_designation"/></alternate_designation>"""
)


def _make_alternate_designation(alternate_designation: str) -> FragBuilder:
    return _make_alternate_designation_node(
        {"alternate_designation": alternate_designation}
    )


_make_description_node: NodeBuilderTemplate = interpret_template(
    """<description>
<NODE name="description"/>
      </description>"""
)


def _make_description(description: str) -> FragBuilder:
    return _make_description_node({"description": description})


def _munge(name: str) -> str:
    """Munge the string to act as part of a LID."""
    return name.replace(" ", "_").lower()


def target_identification(
    target_name: str,
    target_type: str,
    alternate_designations: str,
    target_description: str,
    target_lid: str,
    reference_type: str,
) -> NodeBuilder:
    """
    Given a target name and target type, return a function that takes
    a document and returns a filled-out ``<Target_Identification />``
    XML node, used in product labels.
    """
    alternate_designations_list = []
    if len(alternate_designations) != 0:
        alternate_designations_list = alternate_designations.split("\n")
    alternate_designation_nodes: List[NodeBuilder] = [
        _make_alternate_designation(alternate_designation)
        for alternate_designation in alternate_designations_list
    ]

    description_nodes: List[NodeBuilder] = []
    if len(target_description) != 0:
        # properly align multi line textnodes with 8 spaces
        target_description = " " * 8 + target_description
        target_description = target_description.replace("\n", "\n" + " " * 8)
        description_nodes = [_make_description(target_description)]

    func = interpret_template(
        """<Target_Identification>
        <name><NODE name="name"/></name>
        <FRAGMENT name="alternate_designations"/>
        <type><NODE name="type"/></type>
        <FRAGMENT name="description"/>
        <Internal_Reference>
            <lid_reference><NODE name="target_lid"/></lid_reference>
            <reference_type><NODE name="reference_type"/></reference_type>
        </Internal_Reference>
        </Target_Identification>"""
    )(
        {
            "name": target_name,
            "type": target_type,
            "alternate_designations": combine_nodes_into_fragment(
                alternate_designation_nodes
            ),
            "description": combine_nodes_into_fragment(description_nodes),
            "target_lid": target_lid,
            "reference_type": reference_type,
        }
    )
    return func


def target_lid(target_parts: List[str]) -> str:
    target = ".".join(_munge(target_part) for target_part in target_parts)
    return f"urn:nasa:pds:context:target:{target}"


approximate_target_table: Dict[str, List[str]] = {
    "VENUS": ["Venus", "Planet"],
    "MARS": ["Mars", "Planet"],
    "JUP": ["Jupiter", "Planet"],
    "SAT": ["Saturn", "Planet"],
    "URA": ["Uranus", "Planet"],
    "NEP": ["Neptune", "Planet"],
    "CERES": ["Ceres", "Dwarf Planet"],
    "PLU": ["Pluto", "Dwarf Planet"],
    "PLCH": ["Pluto", "Dwarf Planet"],
    "IO": ["Io", "Satellite", "Jupiter"],
    "EUR": ["Europa", "Satellite", "Jupiter"],
    "GAN": ["Ganymede", "Satellite", "Jupiter"],
    "CALL": ["Callisto", "Satellite", "Jupiter"],
    "TITAN": ["Titan", "Satellite", "Saturn"],
    "TRIT": ["Triton", "Satellite", "Neptune"],
    "DIONE": ["Dione", "Satellite", "Saturn"],
    "IAPETUS": ["Iapetus", "Satellite", "Saturn"],
}
