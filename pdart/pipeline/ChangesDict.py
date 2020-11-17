from typing import Dict
from pdart.pds4.LID import LID

CHANGES_DICT = Dict[LID, bool]
CHANGES_DICT_NAME: str = "changes$dict.txt"


def read_changes_dict(changes_path: str) -> CHANGES_DICT:
    changes_dict = dict()
    with open(changes_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                assert len(parts) == 2, parts
                lid, changed = parts
                assert changed in ["False", "True"]
                changes_dict[LID(lid)] = changed == "True"
    return changes_dict


def write_changes_dict(d: CHANGES_DICT, changes_path: str) -> None:
    with open(changes_path, "w") as changes_file:
        for lid, changed in sorted(d.items()):
            print(lid, changed, file=changes_file)
