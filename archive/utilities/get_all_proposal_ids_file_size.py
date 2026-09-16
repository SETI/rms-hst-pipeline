# type: ignore
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_ARCHIVE = os.path.join(_REPO_ROOT, 'archive')
_HST = os.path.join(_REPO_ROOT, 'HST')
for _path in (_ARCHIVE, _HST):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from astroquery.mast import Observations
from hst_helper.query_utils import get_filtered_products
from pdart.astroquery.utils import (
    get_table_with_retries,
    ymd_tuple_to_mjd,
)

BYTES_PER_GB = 1024 ** 3
BYTES_PER_TB = 1024 ** 4

if __name__ == "__main__":
    TWD = os.environ["TMP_WORKING_DIR"]
    file_path = f"{TWD}/all_proposal_ids_file_size_count.txt"
    start_date = ymd_tuple_to_mjd((1900, 1, 1))
    end_date = ymd_tuple_to_mjd((2025, 1, 1))

    def mast_call():
        return Observations.query_criteria(
            dataRights="PUBLIC",
            obs_collection=["HST"],
            instrument_name=[],
            t_obs_release=(start_date, end_date),
            mtFlag=True,
        )

    obs = get_table_with_retries(mast_call, 1)
    print(f"Found {len(obs)} observations")
    proposal_ids = list(set(obs["proposal_id"]))
    print(f"Found {len(proposal_ids)} proposal ids")

    products = get_filtered_products(obs)
    print(f"Found {len(products)} accepted products")

    info = {}
    seen_filenames = set()
    for row in products:
        filename = row["productFilename"]
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)
        pid = int(row["proposal_id"])
        size = int(row["size"])
        if pid not in info:
            info[pid] = {"size": 0, "count": 0}
        info[pid]["size"] += size
        info[pid]["count"] += 1

    for pid in sorted(info):
        entry = info[pid]
        size_gb = entry["size"] / BYTES_PER_GB
        print(
            f"proposal id: {pid}, count: {entry['count']} products, "
            f"size: {entry['size']} bytes ({size_gb:.2f} GB)"
        )

    total_size = sum(entry["size"] for entry in info.values())
    total_gb = total_size / BYTES_PER_GB
    total_tb = total_size / BYTES_PER_TB
    print(
        f"Total size: {total_size} bytes ({total_gb:.2f} GB, {total_tb:.2f} TB) "
        f"across {len(info)} proposal ids"
    )

    with open(file_path, "w") as f:
        f.write(
            "==================\n"
            "All proposal ids file size and count\n"
            "==================\n"
        )
        for pid in sorted(info):
            entry = info[pid]
            if entry["count"] == 0 or entry["size"] == 0:
                continue
            size_gb = entry["size"] / BYTES_PER_GB
            f.write(
                f"{pid}: count={entry['count']}, "
                f"size={entry['size']} bytes ({size_gb:.2f} GB)\n"
            )
        f.write(
            f"\nTotal size: {total_size} bytes ({total_gb:.2f} GB, {total_tb:.2f} TB) "
            f"across {len(info)} proposal ids\n"
        )

    print(f"Wrote {len(info)} entries to {file_path}")

# python3 get_all_proposal_ids_file_size.py
