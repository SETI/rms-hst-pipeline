from typing import Any, Dict, List, Set, cast

from fs.base import FS
from fs.memoryfs import MemoryFS
from fs.path import join

from pdart.fs.multiversioned.VersionContents import VersionContents

from pdart.pds4.LIDVID import LIDVID


def write_dictionary_to_fs(fs: FS, dir_path: str, d: Dict[Any, Any]) -> None:
    for k, v in d.items():
        assert type(k) in [str, str]
        type_v = type(v)
        sub_path = join(dir_path, str(k))
        if type_v == dict:
            fs.makedir(sub_path)
            write_dictionary_to_fs(fs, sub_path, v)
        elif type_v in [str, str]:
            fs.writetext(sub_path, str(v))
        else:
            assert False, f"unexpected type {type_v} at {sub_path}"


def dictionary_to_contents(lidvids: Set[LIDVID], d: Dict[Any, Any]) -> VersionContents:
    fs = MemoryFS()
    write_dictionary_to_fs(fs, "/", d)
    filepaths = set(fs.walk.files())
    return VersionContents(True, lidvids, fs, filepaths)
