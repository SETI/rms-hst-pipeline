import os
import os.path
import shutil
import tempfile

from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, \
    create_bundle_db_from_os_filepath


def _bundle_dir(bundle_id, base_directory):
    bundle_name = 'hst_%05d' % bundle_id
    return os.path.join(base_directory, bundle_name)


def create_bundle_dir(bundle_id, base_directory):
    # type: (int, unicode) -> None
    bundle_dir = _bundle_dir(bundle_id, base_directory)
    if os.path.isdir(bundle_dir):
        # handle "it already exists" case
        pass
    elif os.path.isfile(bundle_dir):
        raise Exception('intended base directory %s exists and is a file' % \
                        bundle_dir)
    else:
        os.mkdir(bundle_dir)


def create_bundle_db(bundle_id, base_directory):
    # type: (unicode) -> BundleDB
    bundle_dir = _bundle_dir(bundle_id, base_directory)
    db = create_bundle_db_from_os_filepath(os.path.join(bundle_dir,
                                                        _BUNDLE_DB_NAME))
    db.create_tables()
    return db


if __name__ == '__main__':
    base_directory = tempfile.mkdtemp()
    create_bundle_dir(12345, base_directory)
    create_bundle_dir(54321, base_directory)
    shutil.rmtree(base_directory)
