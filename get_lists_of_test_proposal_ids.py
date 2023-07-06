# type: ignore
import os
import sys

from pdart.astroquery.astroquery import (
    MastSlice,
    CustomizedQueryMastSlice,
    ProductSet,
)
from astroquery.mast import Observations
from pdart.astroquery.utils import (
    filter_table,
    get_table_with_retries,
    ymd_tuple_to_mjd,
)
from pdart.pipeline.suffix_info import INSTRUMENT_FROM_LETTER_CODE  # type: ignore

from pdart.pipeline.directories import DevDirectories

# query without image data product type
INST_LI = [
    "COS/NUV",
    "STIS/CCD",
    "STIS/NUV-MAMA",
    "WFPC2/PC",
    "WFPC2/WFC",
    "STIS/FUV-MAMA",
    "HRS/1",
    "ACS/HRC",
    "COS/FUV",
    "HRS/2",
    "FOS/RD",
    "FOS/BL",
    "WFPC/PC",
    "WFPC/WFC",
    "NICMOS/NIC1",
    "NICMOS/NIC3",
    "NICMOS/NIC2",
    "ACS/SBC",
    "ACS/WFC",
    "FOC/48",
    "FOC/96",
    "HSP/VIS/PMT",
    "HSP/UNK/VIS",
    "HSP/UNK/POL",
    "HSP/UNK/PMT",
    "HRS",
    "WFC3/UVIS",
    "WFC3/IR",
]
# query with image data product type
# IMAGE_INST_LI = ["ACS/SBC"]
IMAGE_INST_LI = [
    "COS/NUV",
    "STIS/CCD",
    "STIS/FUV-MAMA",
    "WFPC2/PC",
    "WFPC2/WFC",
    "NICMOS/NIC1",
    "NICMOS/NIC2",
    "ACS/HRC",
    "STIS/NUV-MAMA",
    "WFPC/PC",
    "WFPC/WFC",
    "WFC3/UVIS",
    "WFC3/IR",
    "ACS/WFC",
    "FOC/96",
    "FOC/48",
    "NICMOS/NIC3",
    "ACS/SBC",
]
# Download All files from Mast
if __name__ == "__main__":
    TWD = os.environ["TMP_WORKING_DIR"]
    assert len(sys.argv) <= 3, sys.argv

    file_path = f"{TWD}/test_proposal_ids_lists.txt"
    start_date = ymd_tuple_to_mjd((1900, 1, 1))
    end_date = ymd_tuple_to_mjd((2025, 1, 1))
    try:
        product_type = str(sys.argv[1])
    except IndexError:
        assert False, "Please specify product type: -image or -n"
    if product_type == "-image":
        target_inst = IMAGE_INST_LI
    elif product_type == "-n":
        target_inst = INST_LI
    dict = {}
    filter_dict = {}
    for inst in target_inst:
        obs = Observations.query_criteria(
            # dataproduct_type=["image"],
            dataRights="PUBLIC",
            obs_collection=["HST"],
            instrument_name=inst,
            t_obs_release=(start_date, end_date),
            mtFlag=True,
        )
        proposal_ids = list(set(obs["proposal_id"]))
        info = {}
        sum_size = 0
        sum_count = 0
        cnt = 0
        for id in proposal_ids:
            id = int(id)
            slice = CustomizedQueryMastSlice((1900, 1, 1), (2025, 1, 1), id)
            try:
                product_set = slice.to_product_set(id)
            except:
                print(f"{id} has empty observation========")
                continue
            count = product_set.product_count()
            size = product_set.download_size()
            if count == 0 or size == 0:
                continue
            sum_size += int(size)
            sum_count += int(count)
            cnt += 1
            print(f"proposal id: {id}, count: {count} products, size: {size} bytes")
            info[id] = {"size": size, "count": count}
        if sum_size == 0 or sum_count == 0 or cnt == 0:
            continue
        avg_size = int(sum_size / cnt)
        avg_count = int(sum_count / cnt)
        filter_li = []
        for k, v in info.items():
            # constraints for test ids
            if v["size"] <= avg_size and v["count"] < 500:
                filter_li.append(k)
        filter_dict[inst] = filter_li
        info["avg_files_size"] = avg_size
        info["avg_files_count"] = avg_count
        dict[inst] = info
    # print(dict)
    print(filter_dict)

    with open(file_path, "w") as f:

        f.write(
            "==================\nProposal ids for pipeline testing\n==================\n"
        )
        for k, v in filter_dict.items():
            f.write("%s: %s\n" % (k, v))
        f.write(
            "==================\nFull files count and size info\n==================\n"
        )
        for k, v in dict.items():
            f.write("%s:\n" % k)
            for id, info in v.items():
                f.write("%s: %s\n" % (id, info))

# Get the ids for testing
# python3 get_lists_of_test_proposal_ids.py -image
# python3 get_lists_of_test_proposal_ids.py -n
