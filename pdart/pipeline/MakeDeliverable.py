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
    ) as version_view, make_sv_deltas(
        version_view, archive_primary_deltas_dir
    ) as sv_deltas, make_sv_deltas(
        sv_deltas, archive_browse_deltas_dir
    ) as browse_deltas, make_sv_deltas(
        browse_deltas, archive_label_deltas_dir
    ) as label_deltas:
        # Hack-ish: just trying to get everything into place
        deliverable_path = fs.path.join(working_dir, 'deliverable')
        os.mkdir(deliverable_path)
        deliverable_fs = DeliverableFS(OSFS(deliverable_path))
        fs.copy.copy_fs(label_deltas, deliverable_fs)
        
        # instrumentation
        deliverable_fs.tree()

