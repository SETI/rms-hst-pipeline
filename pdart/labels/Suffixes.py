from typing import Generator, List

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID


"""The suffixes considered raw data, in order of preference."""
RAW_SUFFIXES: List[str] = [
    "raw",
    "flt",
    "drz",
    "crj",
    "d0m",
    # "d0f", # waivered
    "c0m",
    # "c0f" # waivered
]

"""The suffixes used to extract Hst_Parameter information."""
SHM_SUFFIXES: List[str] = ["shm", "spt"]
