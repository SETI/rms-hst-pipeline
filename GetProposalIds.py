import os
import sys

from pdart.astroquery.Astroquery import (
    MastSlice,
    CustomizedQueryMastSlice,
)

# Download from Mast
if __name__ == "__main__":
    assert len(sys.argv) == 2, sys.argv

    product_type = str(sys.argv[1])
    if product_type == "all":
        # Get the list of proposal ids with moving target = true
        list_dir = f"./proposal_ids_all.txt"
        customized_query_slice = CustomizedQueryMastSlice((1900, 1, 1), (2025, 1, 1))
        proposal_ids_list = customized_query_slice.get_proposal_ids()
    elif product_type == "image":
        # Get the list of proposal ids with image product type & moving target = true
        list_dir = f"./proposal_ids_image.txt"
        slice = MastSlice((1900, 1, 1), (2025, 1, 1))
        proposal_ids_list = slice.get_proposal_ids()
    else:
        assert False, "Invalid command for GetProposalIds.py"

    # Write proposal ids into proposal_ids.txt
    with open(list_dir, "w") as f:
        for proposal_id in proposal_ids_list:
            f.write("%s\n" % proposal_id)
