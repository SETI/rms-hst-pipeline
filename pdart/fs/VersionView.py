from pdart.fs.ReadOnlyView import ReadOnlyView

from fs.path import join

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fs.base import FS


class VersionView(ReadOnlyView):
    def __init__(self, bundle_lidvid, wrap_fs):
        # type: (unicode, FS) -> None
        bundle_lid, self.version_id = bundle_lidvid.split(u'::')[-2:]
        self.bundle_id = bundle_lid.split(u':')[-1]
        assert wrap_fs.exists(join(u'/', self.bundle_id,
                                   u'v%s' % self.version_id))
        ReadOnlyView.__init__(self, wrap_fs)
