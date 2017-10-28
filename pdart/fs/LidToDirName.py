from fs.path import join

from pdart.pds4.LID import LID


def lid_to_dir_name(lid):
    # type: (LID) -> unicode
    parts = filter(lambda (x): x is not None,
                   [u'/', lid.bundle_id, lid.collection_id, lid.product_id])
    return apply(join, parts)
