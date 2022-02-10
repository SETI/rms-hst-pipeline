import os
import sys

from pdart.astroquery.astroquery import (
    MastSlice,
    CustomizedQueryMastSlice,
    ProductSet,
)
from pdart.pipeline.suffix_info import INSTRUMENT_FROM_LETTER_CODE  # type: ignore

from pdart.pipeline.directories import DevDirectories

# Download All files from Mast
if __name__ == "__main__":
    TWD = os.environ["TMP_WORKING_DIR"]
    assert len(sys.argv) <= 3, sys.argv
    proposal_id = int(sys.argv[1])
    action = str(sys.argv[2]) if len(sys.argv) == 3 else None
    path = f"{TWD}/files_from_mast"

    dir = DevDirectories(path)
    working_dir = dir.working_dir(proposal_id)
    file_path = f"{working_dir}/{proposal_id:05}_files_info.txt"

    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    slice = CustomizedQueryMastSlice((1900, 1, 1), (2025, 1, 1), proposal_id)
    result = slice.get_products(proposal_id)

    # Get full list of files and unique suffixes of one proposal id
    suffixes = []
    inst = ""
    with open(file_path, "w") as f:

        f.write("==================\nList of files:\n==================\n")
        for row in result:
            productFilename = row["productFilename"]
            # Using iterrows is faster but mypy will complain about Table dosen't
            # have attribute iterrows. We can use # type: ignore to ignore it.
            # for (productFilename, productSubGroupDescription) in result.iterrows(  # type: ignore
            #     "productFilename", "productSubGroupDescription"
            # ):
            if action != "-s":
                f.write("%s\n" % productFilename)
            if not inst:
                inst = INSTRUMENT_FROM_LETTER_CODE[productFilename[0]]
            productSubGroupDescription = row["productSubGroupDescription"]
            suffixes.append(str(productSubGroupDescription))
        unique_suffixes = set(suffixes)
        f.write("==================\nInstrument:\n==================\n%s\n" % inst)
        f.write(
            "==================\nUnique suffixes:\n==================\n%s\n"
            % unique_suffixes
        )

    # Download all files
    if action == "-d":
        product_set = ProductSet(result)
        product_set.download(working_dir)
# Get the suffixes for one proposal id:
# python3 download_all_files_for_one_proposal_id.py 07885 -s
