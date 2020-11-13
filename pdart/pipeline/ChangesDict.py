from typing import Dict
from pdart.pds4.LID import LID

CHANGES_DICT: str = "changes$dict.txt"


def read_changes_dict(changes_path: str) -> Dict[LID, bool]:
    changes_dict = dict()
    with open(changes_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                assert len(parts) == 2, parts
                lid, changed = parts
                assert changed in ["False", "True"]
                changes_dict[LID(lid)] = bool(changed)
    return changes_dict


def write_changes_dict(d: Dict[LID, bool], changes_path: str) -> None:
    with open(changes_path, "w") as changes_file:
        for lid, changed in d.items():
            print(lid, changed, file=changes_file)
