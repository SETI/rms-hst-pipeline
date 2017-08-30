from typing import TYPE_CHECKING
from fs.path import join

if TYPE_CHECKING:
    from fs.base import FS


_SUBDIR_VERSIONS_FILENAME = u'subdir$versions.txt'
# type: unicode


def parseSubdirVersions(txt):
    # type: (unicode) -> Dict[unicode, unicode]
    d = {}
    for n, line in enumerate(txt.split('\n')):
        line = line.strip()
        if line:
            fields = line.split(' ')
            assert len(fields) is 2, "line #%d = %r" % (n, line)
            # TODO assert format of each field
            d[fields[0]] = fields[1]
    return d


def strSubdirVersions(d):
    # type: (Dict[unicode, unicode]) -> unicode
    return ''.join(['%s %s\n' % (k, v) for k, v in sorted(d.items())])


def readSubdirVersions(fs, dir):
    # type: (FS, unicode) -> Dict[unicode, unicode]
    SUBDIR_VERSIONS_FILEPATH = join(dir, _SUBDIR_VERSIONS_FILENAME)
    return parseSubdirVersions(fs.gettext(SUBDIR_VERSIONS_FILEPATH))


def writeSubdirVersions(fs, dir, d):
    # type: (FS, unicode, Dict[unicode, unicode]) -> None
    SUBDIR_VERSIONS_FILEPATH = join(dir, _SUBDIR_VERSIONS_FILENAME)
    fs.settext(SUBDIR_VERSIONS_FILEPATH, strSubdirVersions(d))
