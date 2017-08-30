from typing import TYPE_CHECKING
from fs.path import join

from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME
if TYPE_CHECKING:
    from fs.base import FS


def parseSubdirVersions(txt):
    # type: (unicode) -> Dict[unicode, unicode]
    d = {}
    for n, line in enumerate(unicode(txt).split('\n')):
        line = line.strip()
        if line:
            fields = line.split(' ')
            assert len(fields) is 2, "line #%d = %r" % (n, line)
            # TODO assert format of each field
            d[fields[0]] = fields[1]
    return d


def strSubdirVersions(d):
    # type: (Dict[unicode, unicode]) -> unicode
    return unicode(''.join(['%s %s\n' % (k, v) for k, v in sorted(d.items())]))


def readSubdirVersions(fs, dir):
    # type: (FS, unicode) -> Dict[unicode, unicode]
    SUBDIR_VERSIONS_FILEPATH = join(dir, SUBDIR_VERSIONS_FILENAME)
    return parseSubdirVersions(fs.gettext(SUBDIR_VERSIONS_FILEPATH,
                                          encoding='ascii'))


def writeSubdirVersions(fs, dir, d):
    # type: (FS, unicode, Dict[unicode, unicode]) -> None
    SUBDIR_VERSIONS_FILEPATH = join(dir, SUBDIR_VERSIONS_FILENAME)
    fs.settext(SUBDIR_VERSIONS_FILEPATH,
               strSubdirVersions(d),
               encoding='ascii')
