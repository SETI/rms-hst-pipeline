from typing import TYPE_CHECKING, List, cast

from fs.memoryfs import MemoryFS
from fs.path import join

from pdart.fs.multiversioned.VersionContents import VersionContents

if TYPE_CHECKING:
    from typing import Any, Dict, Set
    from fs.base import FS

    from pdart.pds4.LIDVID import LIDVID


def write_dictionary_to_fs(fs, dir_path, d):
    # type: (FS, unicode, Dict[Any, Any]) -> None
    for k, v in d.iteritems():
        assert type(k) in [str, unicode]
        type_v = type(v)
        sub_path = join(dir_path, unicode(k))
        if type_v == dict:
            fs.makedir(sub_path)
            write_dictionary_to_fs(fs, sub_path, v)
        elif type_v in [str, unicode]:
            fs.writetext(sub_path, unicode(v))
        else:
            assert False, "unexpected type %s at %s" % (type_v, sub_path)


def dictionary_to_contents(lidvids, d):
    # type: (Set[LIDVID], Dict[Any, Any]) -> VersionContents
    fs = MemoryFS()
    write_dictionary_to_fs(fs, u"/", d)
    filepaths = set(fs.walk.files())
    return VersionContents(True, lidvids, fs, filepaths)
