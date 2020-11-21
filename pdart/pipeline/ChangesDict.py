from typing import Dict, Optional, Tuple
from pdart.pds4.LID import LID
from pdart.pds4.VID import VID


class ChangesDict(object):
    def __init__(
        self, changes_dict: Optional[Dict[LID, Tuple[VID, bool]]] = None
    ) -> None:
        self.changes_dict = changes_dict if changes_dict else dict()

    def set(self, lid: LID, vid: VID, changed: bool) -> None:
        self.changes_dict[lid] = (vid, changed)

    def vid(self, lid: LID) -> VID:
        return self.changes_dict[lid][0]

    def changed(self, lid: LID) -> bool:
        return self.changes_dict[lid][1]


CHANGES_DICT = ChangesDict
CHANGES_DICT_NAME: str = "changes$dict.txt"


def read_changes_dict(changes_path: str) -> CHANGES_DICT:
    changes_dict = dict()
    with open(changes_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                assert len(parts) == 3, parts
                lid, vid, changed = parts
                assert changed in ["False", "True"]
                changes_dict[LID(lid)] = (VID(vid), changed == "True")
    return ChangesDict(changes_dict)


def write_changes_dict(d: ChangesDict, changes_path: str) -> None:
    with open(changes_path, "w") as changes_file:
        for lid, (vid, changed) in sorted(d.changes_dict.items()):
            print(lid, vid, changed, file=changes_file)
