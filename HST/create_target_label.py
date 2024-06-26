##########################################################################################
# create_target_label.py
#
# Create target label if it doesn't exist in the PDS4 page:
# https://pds.nasa.gov/data/pds4/context-pds4/target/
##########################################################################################

import bs4
from collections import defaultdict
import json
import pdslogger
import urllib

from hst_helper.general_utils import create_collection_label
# from hst_helper.fs_utils import get_deliverable_path

PDS_URL = 'https://pds.nasa.gov/data/pds4/context-pds4/target/'

TARGET_LABEL_TEMPLATE = 'TARGET_LABEL.xml'

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
    target_records = data_dict['target_identifications']
    found_target_list = []
    for targ in target_records:
        # TODO: handle the case for a target body with multiple names or types. Might
        # need a new task for this.
        # name = targ['formatted_name']
        # type = targ['formatted_type']
        lid = targ['lid']
        _, _, target = lid.partition('target:')
        if target not in found_target_list:
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
            label_filename = f'{target}_1.0.xml'
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
    """Create tmp-context-products.json to store additional context product information
    used for validation. It will be passed to --add-context-products parameter in the
    validation, for development purpose only.

    Inputs:
        proposal_id    a proposal id.
        data_dict      a data dictionary used to create the label.
    """
    # TODO: figure out the version id for all context data
    vid = '1.0'
    context_data = [
        f'urn:nasa:pds:context:investigation:individual.hst_{proposal_id:05}::{vid}',
        f'urn:nasa:pds:context:instrument_host:spacecraft.hst::{vid}'
    ]

    for inst in data_dict['inst_id_li']:
        context_data.append(f'urn:nasa:pds:context:instrument:hst.{inst.lower()}::{vid}')

    json_data = defaultdict(list)
    for data in context_data:
        new_context = {
            'name': [],
            'type': [],
            'lidvid': data,
        }
        json_data['Product_Context'].append(new_context)

    # Include newly added target labels
    for targ in targ_list:
        targ_name = targ['formatted_name']
        targ_type = targ['formatted_type']
        targ_lidvid = f"{targ['lid']}::{1.1}"
        new_context = {
            'name': [targ_name],
            'type': [targ_type],
            'lidvid': targ_lidvid,
        }
        json_data['Product_Context'].append(new_context)
    with open(f'tmp-context-products-{proposal_id}.json', 'w') as tmp_json:
        json.dump(json_data, tmp_json)
