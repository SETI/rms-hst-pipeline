"""
Templates to create an ``<Observing_System />`` XML element.
Roll-up: One <Observing_System /> for bundle, data colleciton, and data.
There can be multiple instruments for a bundle, but only one instrument for
data collection and data.
"""

from typing import Dict, List
from pdart.xml.Templates import (
    combine_nodes_into_fragment,
    FragBuilder,
    NodeBuilder,
    NodeBuilderTemplate,
    interpret_template,
)

_observing_system: NodeBuilderTemplate = interpret_template(
    """<Observing_System>
      <name><NODE name="name"/></name>
      <Observing_System_Component>
        <name>Hubble Space Telescope</name>
        <type>Host</type>
        <Internal_Reference>
          <lid_reference><NODE name="instrument_host_lid"/></lid_reference>
          <reference_type>is_instrument_host</reference_type>
        </Internal_Reference>
      </Observing_System_Component>
      <FRAGMENT name="observing_instrument"/>
    </Observing_System>"""
)
"""
An interpreted fragment template to create an ``<Observing_System />``
XML element.
"""

_make_observing_instrument_node: NodeBuilderTemplate = interpret_template(
    """<Observing_System_Component>
      <name><NODE name="component_name"/></name>
      <type>Instrument</type>
      <Internal_Reference>
        <lid_reference><NODE name="instrument_lid"/></lid_reference>
        <reference_type>is_instrument</reference_type>
      </Internal_Reference>
    </Observing_System_Component>"""
)


def _make_observing_instrument(instrument_info: Dict) -> FragBuilder:
    return _make_observing_instrument_node(
        {
            "component_name": instrument_info["component_name"],
            "instrument_lid": instrument_info["instrument_lid"],
        }
    )


def get_host_name(instrument: str) -> str:
    return {
        "acs": "Hubble Space Telescope Advanced Camera for Surveys",
        "wfc3": "Hubble Space Telescope Wide Field Camera 3",
        "wfpc2": "Hubble Space Telescope Wide-Field Planetary Camera 2",
        "nicmos": "Hubble Space Telescope Near Infrared Camera and Multi-Object Spectrometer",
        "foc": "Hubble Space Telescope Faint Object Camera",
        "cos": "Hubble Space Telescope Cosmic Origins Spectrograph",
        "wfpc": "Hubble Space Telescope Wide-Field Planetary Camera 1",
        "stis": "Hubble Space Telescope Space Telescope Imaging Spectrograph",
        "fos": "Hubble Space Telescope Faint Object Spectrograph",
        "ghrs": "Hubble Space Telescope Faint Object Spectrograph",
    }[instrument]


def get_instrument_component_name(instrument: str) -> str:
    # Same as get_component_host_name(instrument)[23:]
    return {
        "acs": "Advanced Camera for Surveys",
        "wfc3": "Wide Field Camera 3",
        "wfpc2": "Wide-Field Planetary Camera 2",
        "nicmos": "Near Infrared Camera and Multi-Object Spectrometer",
        "foc": "Faint Object Camera",
        "cos": "Cosmic Origins Spectrograph",
        "wfpc": "Wide-Field Planetary Camera 1",
        "stis": "Space Telescope Imaging Spectrograph",
        "fos": "Faint Object Spectrograph",
        "ghrs": "Faint Object Spectrograph",
    }[instrument]


def observing_instrument_dict(instrument: str) -> Dict:
    return {
        "component_name": get_instrument_component_name(instrument),
        "instrument_lid": observing_system_lid(instrument),
    }


def observing_system_lid(instrument: str) -> str:
    return f"urn:nasa:pds:context:instrument:insthost.{instrument}"


def instrument_host_lid() -> str:
    return "urn:nasa:pds:context:instrument_host:spacecraft.hst"


def observing_system(instruments: List[str]) -> NodeBuilder:
    """
    Given a list of instruments, return an interpreted fragment template to
    create an ``<Observing_System />`` XML element.
    """
    observing_instrument_nodes: List[NodeBuilder] = []
    host_name = ""
    for instrument in instruments:
        # Get the host name from the 1st instrument
        if not host_name:
            host_name = get_host_name(instrument)
        inst_info = {
            "component_name": get_instrument_component_name(instrument),
            "instrument_lid": observing_system_lid(instrument),
        }
        _make_observing_instrument(inst_info)
        observing_instrument_nodes.append(_make_observing_instrument(inst_info))

    return _observing_system(
        {
            "name": host_name,
            "instrument_host_lid": instrument_host_lid(),
            "observing_instrument": combine_nodes_into_fragment(
                observing_instrument_nodes
            ),
        }
    )
