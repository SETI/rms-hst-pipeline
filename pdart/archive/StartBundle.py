import os
import os.path
import shutil
import tempfile

def create_bundle_dir(bundle_id, base_directory):
    # type: (int, unicode) -> None
    bundle_name = 'hst_%05d' % bundle_id
    bundle_dir = os.path.join(base_directory, bundle_name)
    if os.path.isdir(bundle_dir):
        # handle "it already exists" case
        pass
    elif os.path.isfile(bundle_dir):
        raise Error('intended base directory %s exists and is a file' % \
                        bundle_dir)
    else:
        os.mkdir(bundle_dir)


if __name__ == '__main__':
    base_directory = tempfile.mkdtemp()
    create_bundle_dir(12345, base_directory)
    create_bundle_dir(54321, base_directory)
    shutil.rmtree(base_directory)
