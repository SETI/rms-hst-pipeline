##########################################################################################
# create_target_label.py
#
# Create target label if it doesn't exist in the PDS4 page:
# https://pds.nasa.gov/data/pds4/context-pds4/target/
##########################################################################################

import bs4
from collections import defaultdict
import json
import os
import pdslogger
import urllib.request
from xml.etree import ElementTree

from hst_pipeline.hst_helper.fs_utils import (get_context_products_path,
                                              get_deliverable_path)
from hst_pipeline.hst_helper.general_utils import create_collection_label
from hst_pipeline.product_labels.suffix_info import INSTRUMENT_NAMES, LABEL_SUFFIX

PDS_URL = 'https://pds.nasa.gov/data/pds4/context-pds4/target/'

TARGET_LABEL_TEMPLATE = 'TARGET_LABEL' + LABEL_SUFFIX
TARGET_LABEL_VID = '1.0'     # the version_id in TARGET_LABEL.lblx

# Names and types of the context products, exactly as the label templates write them
# (PRODUCT_LABEL.lblx, PRODUCT_COLLECTION_LABEL.lblx, BUNDLE_LABEL.lblx), so that validate
# can match each context reference against the temporary context-products JSON.
INVESTIGATION_NAME_FMT = 'HST observing program {}'
INVESTIGATION_TYPE = 'Individual Investigation'
HOST_NAME = 'Hubble Space Telescope'
HOST_TYPE = 'Host'
INSTRUMENT_TYPE = 'Instrument'
INSTRUMENT_LID_PREFIX = 'urn:nasa:pds:context:instrument:hst.'

def create_target_label(proposal_id, data_dict, logger=None):
    """Create target label under context collection if it doesn't exist in PDS page.

    Inputs:
        proposal_id    a proposal id.
        data_dict      a data dictionary used to create the label.
        logger         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Create target label for proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Download the list of existing PDS4 target context product labels into pds_targ_list
    with urllib.request.urlopen(PDS_URL) as response:
        html = response.read()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    a_tags = soup.find_all('a')
    pds_targ_list = [a.string for a in a_tags if a.string]

    # Get the target info (name & type) from the passed in data dict, and check to see
    # if it exists in the pds_targ_list (from PDS page).
    # A target can appear more than once, identified with different alternate
    # designations; it needs only one label.
    target_records = data_dict['target_identifications']
    found_target_list = []
    found_lids = set()
    for targ in target_records:
        lid = targ['lid']
        _, _, target = lid.partition('target:')
        if lid not in found_lids:
            found_lids.add(lid)
            found_target_list.append((targ, target))

    # Compare target info from data dict to the ones from PDS page.
    missing_target_list = []
    for entry in found_target_list:
        target_info = entry[0]
        target = entry[1]
        is_target_label_exists = False
        for label_name in pds_targ_list:
            if target in label_name:
                is_target_label_exists = True
                break
        # Create target label under context folder if it doesn't exist in PDS page.
        if not is_target_label_exists:
            label_filename = f'{target}_{TARGET_LABEL_VID}{LABEL_SUFFIX}'
            target_data_dict = {
                'target': target_info,
                'label_date': data_dict['label_date']
            }
            create_collection_label(proposal_id, 'context', target_data_dict,
                                    label_filename, TARGET_LABEL_TEMPLATE, logger)
            missing_target_list.append(target_info)

    # Include newly created target labels in tmp-context-products.json, this is to
    # include unregistered context in validation during development.
    create_tmp_context_json(proposal_id, data_dict, missing_target_list)

def create_tmp_context_json(proposal_id, data_dict, targ_list=[]):
    """Create tmp-context-products-<nnnnn>.json to store additional context product
    information used for validation. It is written beside the bundle (see
    get_context_products_path) and passed to the --add-context-products parameter in the
    validation, for development purpose only.

    Besides the program's own instruments, it lists every HST instrument that a handbook
    label in the bundle's document collection refers to, since a handbook covering
    several instruments (e.g., Volume II of the HST Data Handbook) refers to all of them.

    Inputs:
        proposal_id    a proposal id.
        data_dict      a data dictionary used to create the label.
        targ_list      the target dictionaries of newly created target labels.
    """
    # TODO: figure out the version id for all context data
    vid = '1.0'
    context_data = [
        (f'urn:nasa:pds:context:investigation:individual.hst_{proposal_id:05}::{vid}',
         INVESTIGATION_NAME_FMT.format(proposal_id), INVESTIGATION_TYPE),
        (f'urn:nasa:pds:context:instrument_host:spacecraft.hst::{vid}',
         HOST_NAME, HOST_TYPE),
    ]

    inst_ids = list(data_dict['inst_id_li'])
    document_dir = os.path.join(get_deliverable_path(proposal_id), 'document')
    for inst in handbook_instruments(document_dir):
        if inst not in inst_ids:
            inst_ids.append(inst)

    for inst in inst_ids:
        context_data.append(
            (f'urn:nasa:pds:context:instrument:hst.{inst.lower()}::{vid}',
             INSTRUMENT_NAMES[inst], INSTRUMENT_TYPE))

    json_data = defaultdict(list)
    for (lidvid, name, type_) in context_data:
        new_context = {
            'name': [name],
            'type': [type_],
            'lidvid': lidvid,
        }
        json_data['Product_Context'].append(new_context)

    # Include newly added target labels, with the name, alternate designations, type,
    # and version that TARGET_LABEL.lblx writes, since the product labels refer to the
    # target by its name and type.
    for targ in targ_list:
        names = [targ['name']]
        for alt in targ.get('alternate_designations', []):
            if alt not in names:
                names.append(alt)
        new_context = {
            'name': names,
            'type': [targ['type']],
            'lidvid': f"{targ['lid']}::{TARGET_LABEL_VID}",
        }
        json_data['Product_Context'].append(new_context)
    json_path = get_context_products_path(proposal_id)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as tmp_json:
        json.dump(json_data, tmp_json)

def handbook_instruments(document_dir):
    """Return the HST instruments that the handbook labels in a document collection
    refer to. A handbook label is a label whose name contains a dash (see
    finalize_document); the instruments are those of its references to
    urn:nasa:pds:context:instrument:hst.<inst> that have an entry in INSTRUMENT_NAMES.

    Input:
        document_dir    the bundle's document collection directory; it need not exist.

    Returns:    a sorted list of instrument ids, e.g., ['FOS', 'GHRS'].
    """
    if not os.path.isdir(document_dir):
        return []

    names = {inst.lower(): inst for inst in INSTRUMENT_NAMES}
    insts = set()
    for label in os.listdir(document_dir):
        if '-' not in label or not label.endswith(LABEL_SUFFIX):
            continue
        root = ElementTree.parse(os.path.join(document_dir, label)).getroot()
        for elem in root.iter():
            if elem.tag.rpartition('}')[2] != 'lid_reference':
                continue
            lid = (elem.text or '').strip()
            if lid.startswith(INSTRUMENT_LID_PREFIX):
                inst = names.get(lid[len(INSTRUMENT_LID_PREFIX):])
                if inst:
                    insts.add(inst)
    return sorted(insts)
