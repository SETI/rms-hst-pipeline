from typing import TYPE_CHECKING

import pdart.add_pds_tools
import picmaker  # need to precede this with 'import pdart.add_pds_tools'

import fs.path
import os

from pdart.pds4.LID import LID
from pdart.pipeline.Utils import make_osfs, make_version_view, make_sv_deltas

if TYPE_CHECKING:
    pass

def build_browse(bundle_segment,
                 working_dir,
                 archive_dir,
                 archive_primary_deltas_dir,
                 archive_browse_deltas_dir):
    # type: (str, unicode, unicode, unicode, unicode) -> None
    with make_osfs(archive_dir) as archive_osfs, \
            make_version_view(archive_osfs, bundle_segment) as version_view, \
            make_sv_deltas(version_view, archive_primary_deltas_dir) as sv_deltas, \
            make_sv_deltas(sv_deltas, archive_browse_deltas_dir) as browse_deltas:
        bundle_path = u'/%s$/' % bundle_segment
        collection_segments = [coll[:-1]
                               for coll
                               in browse_deltas.listdir(bundle_path)
                               if '$' in coll]
        for collection_segment in collection_segments:
            if collection_segment.startswith('data_'):
                data_lid = LID.create_from_parts([bundle_segment,
                                                  collection_segment])
                browse_lid = data_lid.to_browse_lid()
                collection_path = u'%s%s$/' % (bundle_path, collection_segment)
                browse_collection_path = u'%s%s$/' % (bundle_path,
                                                      browse_lid.collection_id)
                # TODO Here I should also add to database, and it
                # needs to be layered.
                browse_deltas.makedirs(browse_collection_path, recreate=True)
                product_segments = [prod[:-1]
                                    for prod
                                    in browse_deltas.listdir(collection_path)
                                    if '$' in prod]
                for product_segment in product_segments:
                    product_path = u'%s%s$/' % (collection_path,
                                                product_segment)
                    browse_product_path = u'%s%s$/' % (browse_collection_path,
                                                       product_segment)
                    print '**** browse product:', browse_product_path
                    # TODO Here I should also add to database, and it
                    # needs to be layered.
                    browse_deltas.makedirs(browse_product_path, recreate=True)
                    for fits_file in browse_deltas.listdir(product_path):
                        fits_filepath = fs.path.join(product_path, fits_file)
                        fits_os_filepath = browse_deltas.getsyspath(
                            fits_filepath)

                        browse_file = fs.path.splitext(fits_file)[0] + '.jpg'
                        browse_filepath = fs.path.join(browse_product_path,
                                                       browse_file)

                        # In a COWFS, a directory does not have a
                        # syspath, only files.  So we write a stub
                        # file into the directory, find its syspath
                        # and its directory's syspath.  Then we remove
                        # the stub file.
                        browse_deltas.touch(browse_filepath)
                        browse_product_os_filepath = browse_deltas.getsyspath(
                            browse_filepath)
                        browse_deltas.remove(browse_filepath)

                        browse_product_os_dirpath = fs.path.dirname(
                            browse_product_os_filepath)
                        
                        # Picmaker expects a list of strings.  If you give it
                        # unicode, it'll index into it and complain about '/'
                        # not being a file.  So don't do that!
                        picmaker.ImagesToPics([str(fits_os_filepath)],
                                              browse_product_os_dirpath,
                                              filter="None",
                                              percentiles=(1, 99))
                        browse_os_filepath = fs.path.join(
                            browse_product_os_dirpath, browse_file)
                        size = os.stat(browse_os_filepath).st_size

        browse_deltas.tree()

