import os.path


def parent_dir(path):
    if path == '/':
        return None
    if path[-1] == '/':
        path = path[:-1]
    (parent, basename) = os.path.split(path)
    return parent
