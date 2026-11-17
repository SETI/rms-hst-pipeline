##########################################################################################
# label_browse_products.py
#
# Create Product_Browse XML labels for browse_* collections of MAST previews.
# One label per IPPPSSOOT (inventory member), with one File_Area_Browse per image file.
##########################################################################################

import datetime
import os
from collections import defaultdict

import pdslogger

from hst_pipeline.hst_helper import BROWSE_PROD_EXT
from hst_pipeline.hst_helper.fs_utils import (file_md5,
                                              get_file_suffix,
                                              get_format_term,
                                              get_formatted_proposal_id,
                                              get_instrument_id_from_fname)
from hst_pipeline.hst_helper.general_utils import create_xml_label
from hst_pipeline.product_labels.suffix_info import browse_data_collection_name

LABEL_VERSION = '1.0'
LABEL_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
TEMPLATE_NAME = 'BROWSE_PRODUCT_LABEL.xml'
VID = (1, 0)

_ENCODING = {
    'jpg': 'JPEG',
    'jpeg': 'JPEG',
    'png': 'PNG',
}


def _file_entry(filepath):
    filename = os.path.basename(filepath)
    _, _, ext = filename.rpartition('.')
    encoding = _ENCODING.get(ext.lower())
    if encoding is None:
        raise ValueError(f'Unsupported browse image type: {filename}')
    timetag = os.path.getmtime(filepath)
    creation = (datetime.datetime.utcfromtimestamp(timetag)
                .strftime('%Y-%m-%dT%H:%M:%SZ'))
    return {
        'file_name': filename,
        'creation_date_time': creation,
        'md5_checksum': file_md5(filepath),
        'object_length': os.path.getsize(filepath),
        'encoding_standard_id': encoding,
    }


def label_browse_collection_directory(collection_dir, proposal_id, logger=None,
                                      version_id=VID):
    """Write Product_Browse labels for all browse images under one collection directory.

    Groups image files by IPPPSSOOT (format_term) within each visit_* subdirectory and
    writes {ipppssoot}.xml next to those images.

    Inputs:
        collection_dir    path to a browse_* directory.
        proposal_id       proposal id (int or str).
        logger            pdslogger to use; None for default EasyLogger.
        version_id        (major, minor) product version tuple.

    Returns:    number of browse product labels written.
    """
    logger = logger or pdslogger.EasyLogger()
    collection_name = os.path.basename(os.path.normpath(collection_dir))
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'templates', TEMPLATE_NAME)

    labels_written = 0
    for root, _, files in os.walk(collection_dir):
        by_ipppssoot = defaultdict(list)
        for name in files:
            _, _, ext = name.rpartition('.')
            if ext.lower() not in BROWSE_PROD_EXT:
                continue
            by_ipppssoot[get_format_term(name)].append(os.path.join(root, name))

        for ipppssoot, filepaths in sorted(by_ipppssoot.items()):
            filepaths = sorted(filepaths, key=lambda p: os.path.basename(p).lower())
            # Every preview in one browse product is linked to the same FITS product.
            first_name = os.path.basename(filepaths[0])
            data_collection = browse_data_collection_name(
                get_file_suffix(first_name), get_instrument_id_from_fname(first_name)
            )
            product_lid = (f'urn:nasa:pds:hst_{formatted_proposal_id}'
                           f':{collection_name}:{ipppssoot}')
            data_lidvid = (f'urn:nasa:pds:hst_{formatted_proposal_id}'
                           f':{data_collection}:{ipppssoot}'
                           f'::{version_id[0]}.{version_id[1]}')
            data_dict = {
                'label_version': LABEL_VERSION,
                'label_date': LABEL_DATE,
                'product_lid': product_lid,
                'version_id': version_id,
                'title': (f'Browse image(s) for observation {ipppssoot} from HST '
                          f'Program {int(proposal_id)}.'),
                'modification_date': datetime.datetime.utcnow().strftime('%Y-%m-%d'),
                'data_lidvid': data_lidvid,
                'files': [_file_entry(p) for p in filepaths],
            }
            label_path = os.path.join(root, f'{ipppssoot}.xml')
            create_xml_label(template_path, label_path, data_dict, logger)
            labels_written += 1

    logger.info(f'Wrote {labels_written} browse product label(s) under {collection_dir}')
    return labels_written


def label_browse_products_for_proposal(proposal_id, bundles_or_staging_root, logger=None):
    """Label all browse_* collections under a proposal root (staging or deliverable).

    Inputs:
        proposal_id                 proposal id.
        bundles_or_staging_root     path containing browse_* collection directories
                                    (e.g. .../hst_05167-deliverable or staging program
                                    dir after prepare_browse_products).
        logger                      pdslogger to use; None for default EasyLogger.

    Returns:    total number of browse product labels written.
    """
    logger = logger or pdslogger.EasyLogger()
    total = 0
    for name in sorted(os.listdir(bundles_or_staging_root)):
        if not name.startswith('browse_'):
            continue
        path = os.path.join(bundles_or_staging_root, name)
        if os.path.isdir(path):
            total += label_browse_collection_directory(path, proposal_id, logger)
    return total
