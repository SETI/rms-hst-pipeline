import os
import sys

from pdart.astroquery.Astroquery import (
    MastSlice,
    CustomizedQueryMastSlice,
)

# Download from Mast
if __name__ == "__main__":
    # TWD = os.environ["TMP_WORKING_DIR"]
    assert len(sys.argv) == 2, sys.argv
    # Store the list of proposal ids as a text file in TWD
    # working_dir = TWD
    # if not os.path.isdir(working_dir):
    #     os.makedirs(working_dir)

    product_type = str(sys.argv[1])
    if product_type == "all":
        # Get the list of proposal ids with moving target = true
        list_dir = f"./proposal_ids_all.txt"
        # list_dir = f"{TWD}/proposal_ids_all.txt"
        customized_query_slice = CustomizedQueryMastSlice((1900, 1, 1), (2025, 1, 1))
        proposal_ids_list = customized_query_slice.get_proposal_ids()
    elif product_type == "image":
        # Get the list of proposal ids with image product type & moving target = true
        list_dir = f"./proposal_ids_image.txt"
        # list_dir = f"{TWD}/proposal_ids_image.txt"
        slice = MastSlice((1900, 1, 1), (2025, 1, 1))
        proposal_ids_list = slice.get_proposal_ids()
    else:
        assert False, "Invalid command for GetProposalIds.py"

    # Write proposal ids into proposal_ids.txt
    with open(list_dir, "w") as f:
        count = 0
        for proposal_id in proposal_ids_list:
            f.write("%s\n" % proposal_id)
            count += 1
        # f.write("Total number of proposal ids: %s\n" % count)
