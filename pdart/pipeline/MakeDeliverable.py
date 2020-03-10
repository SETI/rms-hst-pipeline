from fs.osfs import OSFS
from pdart.fs.DeliverableFS import DeliverableFS
import fs.path
import os

from pdart.pipeline.Utils import make_osfs, make_version_view, make_sv_deltas

def make_deliverable(
    proposal_id,
    bundle_segment,
    working_dir,
    archive_dir,
    archive_primary_deltas_dir,
    archive_browse_deltas_dir,
    archive_label_deltas_dir):
    # type: (int, str, unicode, unicode, unicode, unicode, unicode) -> None
    with make_osfs(archive_dir) as archive_osfs, make_version_view(
        archive_osfs, bundle_segment
    ) as version_view:
        # Hack-ish: just trying to get everything into place
        deliverable_path = fs.path.join(working_dir, 'deliverable')
        os.mkdir(deliverable_path)
        deliverable_osfs = OSFS(deliverable_path)
        deliverable_fs = DeliverableFS(deliverable_osfs)
        fs.copy.copy_fs(version_view, deliverable_fs)
        
        # instrumentation
        deliverable_osfs.tree()

