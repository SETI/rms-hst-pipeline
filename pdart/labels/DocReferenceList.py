"""
Templates to create an ``<Reference_List />`` XML element.
Reference_List exists in data product, data collection and bundle labels.
"""
from typing import List

from pdart.xml.Templates import (
    NodeBuilder,
    NodeBuilderTemplate,
    FragBuilder,
    combine_nodes_into_fragment,
    interpret_template,
)


_make_doc_internal_ref_node: NodeBuilderTemplate = interpret_template(
    """<Internal_Reference>
            <lid_reference><NODE name="ref_lid"/></lid_reference>
            <reference_type><NODE name="ref_type"/></reference_type>
            <comment><NODE name="comment"/></comment>
      </Internal_Reference>"""
)


def _make_doc_internal_ref(ref_lid: str, ref_type: str, comment: str) -> FragBuilder:
    return _make_doc_internal_ref_node(
        {
            "ref_lid": ref_lid,
            "ref_type": ref_type,
            "comment": comment,
        }
    )


def get_hst_data_hand_book_lid(instrument: str) -> str:
    return f"urn:nasa:pds:hst-support:document:{instrument}-dhb"


def get_hst_inst_hand_book_lid(instrument: str) -> str:
    return f"urn:nasa:pds:hst-support:document:{instrument}-ihb"


def get_hst_data_hand_book_comment(instrument: str) -> str:
    return f"The Data Handbook for {instrument.upper()}"


def get_hst_inst_hand_book_comment(instrument: str) -> str:
    return f"The Instrument Handbook for {instrument.upper()}"


reference_list: NodeBuilderTemplate = interpret_template(
    """<Reference_List>
      <FRAGMENT name="internal_reference" />
    </Reference_List>"""
)
"""
An interpreted fragment template to create an ``<Reference_List />``
XML element.
"""


def make_document_reference_list(
    instruments: list,
    ref: str,
) -> NodeBuilder:
    """
    Given a list of instruments, return a <Reference_List> node in the label
    """
    internal_reference_nodes: List[NodeBuilder] = []
    for instrument in instruments:
        ref_lid = get_hst_data_hand_book_lid(instrument)
        ref_type = f"{ref}_to_document"
        comment = get_hst_data_hand_book_comment(instrument)
        data_handbook_node = _make_doc_internal_ref(ref_lid, ref_type, comment)
        internal_reference_nodes.append(data_handbook_node)

        ref_lid = get_hst_inst_hand_book_lid(instrument)
        comment = get_hst_inst_hand_book_comment(instrument)
        inst_handbook_node = _make_doc_internal_ref(ref_lid, ref_type, comment)
        internal_reference_nodes.append(inst_handbook_node)

    func = reference_list(
        {
            "internal_reference": combine_nodes_into_fragment(internal_reference_nodes),
        }
    )
    return func
