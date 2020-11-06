import os
import sys
from pdart.astroquery.AcceptedSuffixes import (
    ACCEPTED_SUFFIXES,
    PART_OF_ACCEPTED_SUFFIXES,
)
from pdart.astroquery.Astroquery import MastSlice, ProductSet
from pdart.pipeline.Directories import DevDirectories

# Download from Mast
if __name__ == "__main__":
    TWD = os.environ["TMP_WORKING_DIR"]
    assert len(sys.argv) == 1, sys.argv
    # Store the list of proposal ids as a text file in TWD
    working_dir = TWD
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    list_dir = f"{TWD}/proposal_ids.txt"
    slice = MastSlice((1900, 1, 1), (2025, 1, 1))
    proposal_ids_list = slice.get_proposal_ids()

    # Write proposal ids into proposal_ids.txt
    with open(list_dir, "w") as f:
        count = 0
        for proposal_id in proposal_ids_list:
            f.write("%s\n" % proposal_id)
            count += 1
        f.write("Totol number of proposal ids: %s\n" % count)
