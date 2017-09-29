import re

from fs.path import join
from typing import TYPE_CHECKING

from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME

if TYPE_CHECKING:
    from fs.base import FS
    from typing import Dict

_versionRE = re.compile('^[0-9\.]+$')


def parse_subdir_versions(txt):
    # type: (unicode) -> Dict[unicode, unicode]
    d = {}
    for n, line in enumerate(unicode(txt).split('\n')):
        line = line.strip()
        if line:
            fields = line.split(' ')
            assert len(fields) is 2, "line #%d = %r" % (n, line)
            # TODO assert format of each field
            assert _versionRE.match(str(fields[1]))
            d[fields[0]] = fields[1]
    return d


def str_subdir_versions(d):
    # type: (Dict[unicode, unicode]) -> unicode
    for v in d.itervalues():
        assert _versionRE.match(str(v))
    return unicode(''.join(['%s %s\n' % (k, v) for k, v in sorted(d.items())]))


def read_subdir_versions(fs, dir):
    # type: (FS, unicode) -> Dict[unicode, unicode]
    SUBDIR_VERSIONS_FILEPATH = join(dir, SUBDIR_VERSIONS_FILENAME)
    return parse_subdir_versions(fs.gettext(SUBDIR_VERSIONS_FILEPATH,
                                            encoding='ascii'))


def write_subdir_versions(fs, dir, d):
    # type: (FS, unicode, Dict[unicode, unicode]) -> None
    SUBDIR_VERSIONS_FILEPATH = join(dir, SUBDIR_VERSIONS_FILENAME)
    fs.settext(SUBDIR_VERSIONS_FILEPATH,
               str_subdir_versions(d),
               encoding='ascii')
