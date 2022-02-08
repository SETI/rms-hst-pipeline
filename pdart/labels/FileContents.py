"""
Functionality to build the XML fragment containing the needed
``<Header />`` and ``<Array />`` or ``<Array_2D_Image />`` elements of
a product label using a SQLite database.
"""

from typing import Any, Dict, List

from pdart.db.BundleDB import BundleDB
from pdart.db.FitsFileDB import get_file_offsets
from pdart.labels.FileContentsXml import (
    AXIS_NAME_TABLE,
    BITPIX_TABLE,
    axis_array,
    data_1d_contents,
    data_2d_contents,
    element_array,
    header_contents,
)
from pdart.xml.Templates import (
    FragBuilder,
    NodeBuilder,
    combine_fragments_into_fragment,
    combine_nodes_into_fragment,
)


def _mk_axis_arrays(
    card_dicts: List[Dict[str, Any]], hdu_index: int, axes: int
) -> FragBuilder:
    def mk_axis_array(i: int) -> NodeBuilder:
        axis_name = AXIS_NAME_TABLE[i]

        elements = card_dicts[hdu_index][f"NAXIS{i}"]
        # TODO Check the semantics of sequence_number
        sequence_number = str(i)
        return axis_array(
            {
                "axis_name": axis_name,
                "elements": str(elements),
                "sequence_number": sequence_number,
            }
        )

    return combine_nodes_into_fragment([mk_axis_array(i + 1) for i in range(0, axes)])


def get_file_contents(
    bundle_db: BundleDB,
    card_dicts: List[Dict[str, Any]],
    instrument: str,
    fits_product_lidvid: str,
) -> FragBuilder:
    """
    Given the dictionary of the header fields from a product's FITS
    file, an open connection to the database, and the product's
    :class:`~pdart.pds4.LIDVID`, return an XML fragment containing the
    needed ``<Header />`` and ``<Array />`` or ``<Array_2D_Image />``
    elements for the FITS file's HDUs.
    """

    def get_hdu_contents(
        hdu_index: int, hdrLoc: int, datLoc: int, datSpan: int
    ) -> FragBuilder:
        """
        Return an XML fragment containing the needed ``<Header />``
        and ``<Array />`` or ``<Array_2D_Image />`` elements for the
        FITS file's HDUs.
        """
        local_identifier = f"hdu_{hdu_index}"
        offset = str(hdrLoc)
        object_length = str(datLoc - hdrLoc)
        description = "Global FITS Header" if hdu_index == 0 else "FITS Header"
        header = header_contents(
            {
                "local_identifier": local_identifier,
                "offset": offset,
                "object_length": object_length,
                "description": description,
            }
        )

        if datSpan:
            hdu_card_dict = card_dicts[hdu_index]
            bitpix = int(hdu_card_dict["BITPIX"])
            axes = int(hdu_card_dict["NAXIS"])
            data_type = BITPIX_TABLE[bitpix]
            elmt_arr = element_array({"data_type": data_type})

            if axes not in [1, 2, 3]:
                raise ValueError(
                    f"NAXIS = {axes} in hdu #{hdu_index} in {fits_product_lidvid}."
                )
            if axes == 1:
                data = data_1d_contents(
                    {
                        "offset": str(datLoc),
                        "Element_Array": elmt_arr,
                        "Axis_Arrays": _mk_axis_arrays(card_dicts, hdu_index, axes),
                    }
                )
                node_functions = [header, data]
            elif axes == 2:
                data = data_2d_contents(
                    {
                        "offset": str(datLoc),
                        "Element_Array": elmt_arr,
                        "Axis_Arrays": _mk_axis_arrays(card_dicts, hdu_index, axes),
                    }
                )
                node_functions = [header, data]
            elif axes == 3:
                # "3-D" images from WFPC2 are really four separate 2-D
                # images.  We document them as such.  Well, four or
                # two.  Well, four or two or maybe something else...
                # Aw, we'll say four or two for now.
                if instrument != "wfpc2":
                    raise ValueError(f"NAXIS=3 and instrument={instrument}.")
                naxis3 = int(hdu_card_dict["NAXIS3"])
                if naxis3 not in [2, 4]:
                    raise ValueError(
                        f"NAXIS1={hdu_card_dict['NAXIS1']}, "
                        + f"NAXIS2={hdu_card_dict['NAXIS2']}, "
                        + f"NAXIS3={hdu_card_dict['NAXIS3']}."
                    )

                if datSpan % naxis3 != 0:
                    raise ValueError(f"datSpan={datSpan} & naxis3={naxis3}")
                layerOffset = datSpan // naxis3
                node_functions = [header]
                for n in range(0, naxis3):
                    data = data_2d_contents(
                        {
                            "offset": str(datLoc + n * layerOffset),
                            "Element_Array": elmt_arr,
                            "Axis_Arrays": _mk_axis_arrays(card_dicts, hdu_index, 2),
                        }
                    )
                    node_functions.append(data)
        else:
            node_functions = [header]

        return combine_nodes_into_fragment(node_functions)

    return combine_fragments_into_fragment(
        [
            get_hdu_contents(*hdu)
            for hdu in get_file_offsets(bundle_db, fits_product_lidvid)
        ]
    )
