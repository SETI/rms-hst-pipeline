from typing import TYPE_CHECKING
import traceback

import pdart.add_pds_tools
import picmaker  # need to precede this with 'import pdart.add_pds_tools'

import fs.path
import os

from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, create_bundle_db_from_os_filepath
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Utils import make_osfs, make_version_view, make_sv_deltas

if TYPE_CHECKING:
    from typing import List
    from pdart.new_db.BundleDB import BundleDB
    from pdart.fs.cowfs.COWFS import COWFS

_INITIAL_VID = VID("1.0")  # type: VID


def _create_initial_lidvid_from_parts(parts):
    # type: (List[str]) -> str
    lid = LID.create_from_parts(parts)
    lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(lidvid)


def _extend_initial_lidvid(lidvid, segment):
    # type: (str, str) -> str
    lid = LIDVID(lidvid).lid().extend_lid(segment)
    new_lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(new_lidvid)


def _build_browse_collection(
    db, browse_deltas, bundle_segment, collection_segment, bundle_path
):
    # type: (BundleDB, COWFS, str, str, unicode) -> None
    bundle_lidvid = _create_initial_lidvid_from_parts([bundle_segment])
    data_collection_lidvid = _create_initial_lidvid_from_parts(
        [bundle_segment, collection_segment]
    )
    browse_collection_lid = LIDVID(data_collection_lidvid).lid().to_browse_lid()
    collection_path = u"%s%s$/" % (bundle_path, collection_segment)
    browse_collection_path = u"%s%s$/" % (
        bundle_path,
        browse_collection_lid.collection_id,
    )

    browse_deltas.makedirs(browse_collection_path, recreate=True)
    browse_collection_lidvid = LIDVID.create_from_lid_and_vid(
        browse_collection_lid, _INITIAL_VID
    )
    db.create_non_document_collection(str(browse_collection_lidvid), bundle_lidvid)
    product_segments = [
        str(prod[:-1]) for prod in browse_deltas.listdir(collection_path) if "$" in prod
    ]
    for product_segment in product_segments:
        product_path = u"%s%s$/" % (collection_path, product_segment)
        browse_product_path = u"%s%s$/" % (browse_collection_path, product_segment)
        browse_product_lidvid = _extend_initial_lidvid(
            str(browse_collection_lidvid), product_segment
        )
        fits_product_lidvid = _extend_initial_lidvid(
            data_collection_lidvid, product_segment
        )

        browse_deltas.makedirs(browse_product_path, recreate=True)
        db.create_browse_product(
            browse_product_lidvid, fits_product_lidvid, str(browse_collection_lidvid)
        )
        for fits_file in browse_deltas.listdir(product_path):
            fits_filepath = fs.path.join(product_path, fits_file)
            fits_os_filepath = browse_deltas.getsyspath(fits_filepath)

            browse_file = fs.path.splitext(fits_file)[0] + ".jpg"
            browse_filepath = fs.path.join(browse_product_path, browse_file)

            # In a COWFS, a directory does not have a
            # syspath, only files.  So we write a stub
            # file into the directory, find its syspath
            # and its directory's syspath.  Then we remove
            # the stub file.
            browse_deltas.touch(browse_filepath)
            browse_product_os_filepath = browse_deltas.getsyspath(browse_filepath)
            browse_deltas.remove(browse_filepath)

            browse_product_os_dirpath = fs.path.dirname(browse_product_os_filepath)

            # Picmaker expects a list of strings.  If you give it
            # unicode, it'll index into it and complain about '/'
            # not being a file.  So don't do that!
            try:
                picmaker.ImagesToPics(
                    [str(fits_os_filepath)],
                    browse_product_os_dirpath,
                    filter="None",
                    percentiles=(1, 99),
                )
            except IndexError as e:
                tb = traceback.format_exc()
                message = "File %s: %s\n%s" % (fits_file, str(e), tb)
                raise Exception(message)

            browse_os_filepath = fs.path.join(browse_product_os_dirpath, browse_file)
            size = os.stat(browse_os_filepath).st_size
            db.create_browse_file(
                browse_os_filepath, browse_file, browse_product_lidvid, size
            )


def build_browse(
    bundle_segment,
    working_dir,
    archive_dir,
    archive_primary_deltas_dir,
    archive_browse_deltas_dir,
):
    # type: (str, unicode, unicode, unicode, unicode) -> None
    db_filepath = os.path.join(working_dir, _BUNDLE_DB_NAME)
    db = create_bundle_db_from_os_filepath(db_filepath)

    with make_osfs(archive_dir) as archive_osfs, make_version_view(
        archive_osfs, bundle_segment
    ) as version_view, make_sv_deltas(
        version_view, archive_primary_deltas_dir
    ) as sv_deltas, make_sv_deltas(
        sv_deltas, archive_browse_deltas_dir
    ) as browse_deltas:
        bundle_path = u"/%s$/" % bundle_segment
        collection_segments = [
            str(coll[:-1]) for coll in browse_deltas.listdir(bundle_path) if "$" in coll
        ]
        for collection_segment in collection_segments:
            if collection_segment.startswith("data_"):
                _build_browse_collection(
                    db, browse_deltas, bundle_segment, collection_segment, bundle_path
                )
