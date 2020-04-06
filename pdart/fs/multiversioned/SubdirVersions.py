"""
Functionality to read and write subdir-version dictionaries.  A
subdir-version dictionary has keys corresponding to subdirectory
names, and values corresponding to VIDs.
"""
import re
from typing import Dict, Pattern

from fs.base import FS
from fs.errors import ResourceNotFound
from fs.path import join

SUBDIR_VERSIONS_FILENAME: str = "subdir$versions.txt"

_versionRE: Pattern[str] = re.compile("^[0-9.]+$")


def parse_subdir_versions(txt: str) -> Dict[str, str]:
    """
    Given the (Unicode) contents of a subdir-version file, parse it
    and return a subdir-version dictionary.
    """
    d = {}
    for n, line in enumerate(txt.split("\n")):
        line = line.strip()
        if line:
            fields = line.split(" ")
            assert len(fields) is 2, f"line #{n} = {line!r}"
            # TODO assert format of each field
            assert _versionRE.match(str(fields[1]))
            d[str(fields[0])] = str(fields[1])
    return d


def unparse_subdir_versions(d: Dict[str, str]) -> str:
    """
    Given a subdir-version dictionary, unparse it into a (Unicode)
    string to be stored in a subdir-version file.
    """
    for v in d.values():
        assert _versionRE.match(str(v))
    return "".join([f"{k} {v}\n" for k, v in sorted(d.items())])


def read_subdir_versions_from_directory(fs: FS, dir: str) -> Dict[str, str]:
    """
    Given the path to a directory, return the subdir-version
    dictionary that lives in it.
    """
    SUBDIR_VERSIONS_FILEPATH = join(dir, SUBDIR_VERSIONS_FILENAME)
    try:
        return parse_subdir_versions(
            fs.readtext(SUBDIR_VERSIONS_FILEPATH, encoding="ascii")
        )
    except ResourceNotFound:
        return dict()


def write_subdir_versions_to_directory(fs: FS, dir: str, d: Dict[str, str]) -> None:
    """
    Given the path to a directory, un-parse and write the contents of
    the given subdir-versions dictionary into a subdir-versions file
    in the directory.
    """
    SUBDIR_VERSIONS_FILEPATH = join(dir, SUBDIR_VERSIONS_FILENAME)
    fs.writetext(SUBDIR_VERSIONS_FILEPATH, unparse_subdir_versions(d), encoding="ascii")
