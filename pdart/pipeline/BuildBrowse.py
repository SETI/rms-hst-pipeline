import logging
import os
import traceback
from typing import cast, List, Set

import fs.path
import picmaker

from pdart.astroquery.AcceptedParams import (
    ACCEPTED_SUFFIXES,
    PART_OF_ACCEPTED_SUFFIXES,
)
from pdart.db.BundleDB import (
    BundleDB,
    _BUNDLE_DB_NAME,
    create_bundle_db_from_os_filepath,
)
from pdart.fs.cowfs.COWFS import COWFS
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.ChangesDict import (
    CHANGES_DICT_NAME,
    ChangesDict,
    read_changes_dict,
    write_changes_dict,
)
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view

_LOGGER = logging.getLogger(__name__)

_NON_IMAGE_SUFFIXES: Set[str] = {"ASN", "SHM", "SPT"}

_BROWSE_SUFFIXES: List[str] = [
    suffix for suffix in ACCEPTED_SUFFIXES if suffix not in _NON_IMAGE_SUFFIXES
]


def _requires_browse_collection(collection_segment: str) -> bool:
    parts = collection_segment.lower().split("_")
    return (
        len(parts) == 3 and parts[0] == "data" and parts[2].upper() in _BROWSE_SUFFIXES
    )


def _extend_lidvid(lid: LID, vid: VID, segment: str) -> str:
    new_lid = lid.extend_lid(segment)
    new_lidvid = LIDVID.create_from_lid_and_vid(new_lid, vid)
    return str(new_lidvid)


def _fill_in_old_browse_collection(
    db: BundleDB,
    changes_dict: ChangesDict,
    bundle_lidvid: LIDVID,
    data_collection_lidvid: LIDVID,
) -> None:
    bundle_segment = bundle_lidvid.lid().parts()[0]
    collection_segment = data_collection_lidvid.lid().parts()[1]

    browse_collection_lid = data_collection_lidvid.lid().to_browse_lid()
    browse_collection_segment = browse_collection_lid.collection_id
    browse_collection_vid = data_collection_lidvid.vid()
    browse_collection_lidvid = LIDVID.create_from_lid_and_vid(
        browse_collection_lid, browse_collection_vid
    )

    changes_dict.set(browse_collection_lid, browse_collection_vid, False)
    db.create_bundle_collection_link(str(bundle_lidvid), str(browse_collection_lidvid))
    _LOGGER.info(f"created link and change for {browse_collection_lidvid}")
    for product in db.get_collection_products(str(browse_collection_lidvid)):
        product_lidvid = LIDVID(product.lidvid)
        changes_dict.set(product_lidvid.lid(), product_lidvid.vid(), False)
        _LOGGER.info(f"created link and change for {product_lidvid}")


def _build_browse_collection(
    db: BundleDB,
    changes_dict: ChangesDict,
    browse_deltas: COWFS,
    bundle_lidvid: LIDVID,
    data_collection_lidvid: LIDVID,
    bundle_path: str,
) -> None:
    bundle_segment = bundle_lidvid.lid().parts()[0]
    collection_segment = data_collection_lidvid.lid().parts()[1]

    browse_collection_lid = data_collection_lidvid.lid().to_browse_lid()
    collection_path = f"{bundle_path}{collection_segment}$/"
    browse_collection_segment = browse_collection_lid.collection_id
    browse_collection_path = f"{bundle_path}{browse_collection_segment}$/"
    browse_collection_vid = data_collection_lidvid.vid()
    browse_collection_lidvid = LIDVID.create_from_lid_and_vid(
        browse_collection_lid, browse_collection_vid
    )

    changes_dict.set(browse_collection_lid, browse_collection_vid, True)

    browse_deltas.makedirs(browse_collection_path, recreate=True)

    db.create_other_collection(str(browse_collection_lidvid), str(bundle_lidvid))
    db.create_bundle_collection_link(str(bundle_lidvid), str(browse_collection_lidvid))

    product_segments = [
        str(prod[:-1]) for prod in browse_deltas.listdir(collection_path) if "$" in prod
    ]
    for product_segment in product_segments:
        # These product_segments are from the data_collection
        product_lid = LID.create_from_parts(
            [bundle_segment, collection_segment, product_segment]
        )
        product_vid = changes_dict.vid(product_lid)

        product_path = f"{collection_path}{product_segment}$/"
        browse_product_path = f"{browse_collection_path}{product_segment}$/"

        browse_product_lidvid = _extend_lidvid(
            browse_collection_lid, product_vid, product_segment
        )

        if changes_dict.changed(product_lid):
            fits_product_lidvid = _extend_lidvid(
                data_collection_lidvid.lid(),
                data_collection_lidvid.vid(),
                product_segment,
            )

            bpl = LIDVID(browse_product_lidvid)
            changes_dict.set(bpl.lid(), bpl.vid(), True)

            browse_deltas.makedirs(browse_product_path, recreate=True)
            db.create_browse_product(
                browse_product_lidvid,
                fits_product_lidvid,
                str(browse_collection_lidvid),
            )
            db.create_collection_product_link(
                str(browse_collection_lidvid), browse_product_lidvid
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
                # str, it'll index into it and complain about '/'
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
                    message = f"File {fits_file}: {e}\n{tb}"
                    raise Exception(message)

                browse_os_filepath = fs.path.join(
                    browse_product_os_dirpath, browse_file
                )
                size = os.stat(browse_os_filepath).st_size
                db.create_browse_file(
                    browse_os_filepath, browse_file, browse_product_lidvid, size
                )
        else:
            bpl = LIDVID(browse_product_lidvid)
            changes_dict.set(bpl.lid(), bpl.vid(), False)
            db.create_collection_product_link(
                str(browse_collection_lidvid), browse_product_lidvid
            )


class BuildBrowse(MarkedStage):
    """
    Walk through the data collections and build corresponding browse
    collections for them.
    """

    def _run(self) -> None:
        _LOGGER.info("Entering BuildBrowse.")
        working_dir: str = self.working_dir()
        archive_dir: str = self.archive_dir()
        archive_primary_deltas_dir: str = self.archive_primary_deltas_dir()
        archive_browse_deltas_dir: str = self.archive_browse_deltas_dir()

        assert not os.path.isdir(
            self.deliverable_dir()
        ), "{deliverable_dir} cannot exist for BuildBrowse"

        changes_path = os.path.join(working_dir, CHANGES_DICT_NAME)
        changes_dict = read_changes_dict(changes_path)

        # TODO remove
        changes_dict.dump("BEFORE BuildBrowse")

        db_filepath = os.path.join(working_dir, _BUNDLE_DB_NAME)
        db = create_bundle_db_from_os_filepath(db_filepath)

        bundle_lid = LID.create_from_parts([self._bundle_segment])
        bundle_vid = changes_dict.vid(bundle_lid)
        bundle_lidvid = LIDVID.create_from_lid_and_vid(bundle_lid, bundle_vid)

        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas, make_sv_deltas(
            sv_deltas, archive_browse_deltas_dir
        ) as browse_deltas:
            bundle_path = f"/{self._bundle_segment}$/"
            collection_segments = [
                str(coll[:-1])
                for coll in browse_deltas.listdir(bundle_path)
                if "$" in coll
            ]
            for collection_segment in collection_segments:
                collection_lid = LID.create_from_parts(
                    [self._bundle_segment, collection_segment]
                )
                if _requires_browse_collection(collection_segment):
                    collection_vid = changes_dict.vid(collection_lid)
                    collection_lidvid = LIDVID.create_from_lid_and_vid(
                        collection_lid, collection_vid
                    )
                    if changes_dict.changed(collection_lid):
                        _LOGGER.info(f"making browse for {collection_lidvid}")
                        _build_browse_collection(
                            db,
                            changes_dict,
                            browse_deltas,
                            bundle_lidvid,
                            collection_lidvid,
                            bundle_path,
                        )
                    else:
                        _fill_in_old_browse_collection(
                            db, changes_dict, bundle_lidvid, collection_lidvid
                        )

            write_changes_dict(changes_dict, changes_path)
        _LOGGER.info("Leaving BuildBrowse.")
