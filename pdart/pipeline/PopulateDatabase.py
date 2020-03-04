from typing import TYPE_CHECKING
import os.path

import fs.path

from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, \
    create_bundle_db_from_os_filepath
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Utils import make_osfs, make_sv_deltas, make_version_view

if TYPE_CHECKING:
    from typing import List

_INITIAL_VID = VID('1.0')  # type: VID

def _create_lidvid_from_parts(parts):
    # type: (List[str]) -> str
    lid = LID.create_from_parts(parts)
    lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(lidvid)


def populate_database(bundle_segment,
                      working_dir,
                      archive_dir,
                      archive_primary_deltas_dir):
    # type: (unicode, unicode, unicode, unicode) -> None
    bundle_segment = str(bundle_segment)
    print '**** populate_database(%s, %s) called' % \
        (bundle_segment, working_dir)

    db_filepath = os.path.join(working_dir, _BUNDLE_DB_NAME)
    if os.path.isfile(db_filepath):
        db = create_bundle_db_from_os_filepath(db_filepath)
    else:
        db = create_bundle_db_from_os_filepath(db_filepath)
        db.create_tables()
        bundle_lidvid = _create_lidvid_from_parts([str(bundle_segment)])
        db.create_bundle(bundle_lidvid)
    
    with make_osfs(archive_dir) as archive_osfs, \
        make_version_view(archive_osfs, bundle_segment) as version_view, \
        make_sv_deltas(version_view,
                       archive_primary_deltas_dir)  \
                       as sv_deltas:
        bundle_path = u'/%s$/' % bundle_segment
        collection_segments = [ coll[:-1] 
                                for coll
                                in sv_deltas.listdir(bundle_path)
                                if '$' in coll]
        for collection_segment in collection_segments:
            collection_path = u'%s%s$/' % (bundle_path, collection_segment)
            is_document_collection = collection_segment == u'document'
            print 'collection_path =', collection_path
            collection_lidvid = _create_lidvid_from_parts([bundle_segment,
                                                           collection_segment])
            if is_document_collection:
                db.create_document_collection(collection_lidvid,
                                              bundle_lidvid)
            else:
                db.create_non_document_collection(collection_lidvid,
                                                  bundle_lidvid)
            
            product_segments = [ prod[:-1]
                                 for prod
                                 in sv_deltas.listdir(collection_path)
                                 if '$' in prod ]
            for product_segment in product_segments:
                product_path = u'%s%s$/' % (collection_path, product_segment)
                print 'product_path =', product_path
                product_lidvid = _create_lidvid_from_parts([bundle_segment,
                                                            collection_segment,
                                                            product_segment])
                fits_files = [fits_file
                              for fits_file
                              in sv_deltas.listdir(product_path)
                              if fs.path.splitext(fits_file)[1] == '.fits']
                for fits_file in fits_files:
                    fits_file_path = fs.path.join(product_path, fits_file)
                    db.create_fits_product(product_lidvid,
                                           collection_lidvid)
                    assert False, 'Implement getsyspath()'
                    TODO fits_os_path = sv_deltas.getsyspath(fits_file_path)
                    
                    populate_database_from_fits_file(
                        db,
                        # TODO This doesn't work.  We need a md5 and
                        # need to PyFITS it.
                        fits_os_path,
                        product_lidvid)


    assert db

    assert os.path.isfile(db_filepath), db_filepath
