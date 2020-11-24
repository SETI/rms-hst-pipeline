import os
import sys

from pdart.astroquery.Astroquery import CustomizedQueryMastSlice

# Download All files from Mast
if __name__ == "__main__":
    TWD = os.environ["TMP_WORKING_DIR"]
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    file_path = f"{TWD}/proposal_ids_without_shm_spt.txt"

    slice = CustomizedQueryMastSlice((1900, 1, 1), (2025, 1, 1), proposal_id)
    result = slice.get_products(proposal_id)

    # Log the proposal id that has no SHM and SPT files
    suffixes = []
    unique_suffixes = None

    for row in result:
        productSubGroupDescription = row["productSubGroupDescription"]
        suffixes.append(str(productSubGroupDescription))
    unique_suffixes = set(suffixes)
    with open(file_path, "a") as f:
        if "SHM" not in unique_suffixes and "SPT" not in unique_suffixes:
            formatted_proposal_id = f"{proposal_id:05}"
            f.write("%s\n" % formatted_proposal_id)
