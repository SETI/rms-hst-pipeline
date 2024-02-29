from typing import List

from fs.path import join

from pdart.pds4.lid import LID


def _lid_to_parts(lid: LID) -> List[str]:
    """
    Extract the parts (bundle, collection, product) from the LID and
    return as a list.
    """
    res = [lid.bundle_id]
    if lid.collection_id:
        res.append(lid.collection_id)
    if lid.product_id:
        res.append(lid.product_id)
    return [str(id) for id in res]


def lid_to_dir(lid: LID) -> str:
    """
    Convert a LID to a directory path.
    """
    dir_parts = _lid_to_parts(lid)
    dir_parts.insert(0, "/")
    return join(*dir_parts)
