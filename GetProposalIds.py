from typing import Optional, Tuple
import os
import sys
from pdart.astroquery.AcceptedSuffixes import (
    ACCEPTED_SUFFIXES,
    PART_OF_ACCEPTED_SUFFIXES,
)
from pdart.astroquery.Astroquery import *
from pdart.pipeline.Directories import DevDirectories

_YMD = Tuple[int, int, int]

# Customize MastSlice to use cutomized query criteria
# Don't restrict product type to image
class CustomizedQueryMastSlice(MastSlice):
    def __init__(
        self, start_date: _YMD, end_date: _YMD, proposal_id: Optional[int] = None
    ) -> None:
        super().__init__(start_date, end_date, proposal_id)

        def mast_call() -> Table:
            if proposal_id is not None:
                return Observations.query_criteria(
                    dataRights="PUBLIC",
                    obs_collection=["HST"],
                    proposal_id=str(proposal_id),
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True,
                )
            else:
                return Observations.query_criteria(
                    dataRights="PUBLIC",
                    obs_collection=["HST"],
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True,
                )

        self.observations_table = get_table_with_retries(mast_call, 1)
        self.proposal_ids: Optional[List[int]] = None


# Download from Mast
if __name__ == "__main__":
    TWD = os.environ["TMP_WORKING_DIR"]
    assert len(sys.argv) == 2, sys.argv
    # Store the list of proposal ids as a text file in TWD
    working_dir = TWD
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)

    product_type = str(sys.argv[1])
    if product_type == "all":
        # Get the list of proposal ids with moving target = true
        list_dir = f"{TWD}/proposal_ids_all.txt"
        customized_query_slice = CustomizedQueryMastSlice((1900, 1, 1), (2025, 1, 1))
        proposal_ids_list = customized_query_slice.get_proposal_ids()
    elif product_type == "image":
        # Get the list of proposal ids with image product type & moving target = true
        list_dir = f"{TWD}/proposal_ids_image.txt"
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
        f.write("Total number of proposal ids: %s\n" % count)
