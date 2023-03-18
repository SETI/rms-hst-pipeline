##########################################################################################
# create_target_label/__init__.py
##########################################################################################
from collections import defaultdict
import json
import pdslogger
import bs4
import urllib

from hst_helper.general_utils import create_collection_label
# from hst_helper.fs_utils import get_deliverable_path

PDS_URL = "https://pds.nasa.gov/data/pds4/context-pds4/target/"

TARGET_LABEL_TEMPLATE = 'TARGET_LABEL.xml'

def create_target_label(proposal_id, data_dict, logger=None):
    """Create target lable under context collection if it doesn't exist in PDS page.

    Inputs:
        proposal_id:    a proposal id.
        data_dict:      a data dictionary used to create the label.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()

    logger.info(f'Create target label for proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    with urllib.request.urlopen(PDS_URL) as response:
        html = response.read()
    soup = bs4.BeautifulSoup(html, "html.parser")
    a_tags = soup.find_all("a")
    pds_targ_list = [a.string for a in a_tags if a.string]
    target_records = data_dict['target_identifications']

    target_list = []
    for targ in target_records:
        name = targ['formatted_name']
        type = targ['formatted_type']
        target = f"{type}.{name}".lower()
        if target not in target_list:
            target_list.append((targ, target))

    new_target_context_list = []
    for entry in target_list:
        target_info = entry[0]
        target = entry[1]
        is_target_label_exists = False
        for label in pds_targ_list:
            if target in label:
                is_target_label_exists = True
                break
        # Create target label under context folder if it doesn't exist in PDS page.
        if not is_target_label_exists:
            label_filename = f"{target}_1.0.xml"
            target_data_dict = {
                'target': target_info,
                'label_date': data_dict['label_date']
            }
            create_collection_label(proposal_id, 'context', target_data_dict,
                                    label_filename, TARGET_LABEL_TEMPLATE, logger)
            new_target_context_list.append(target_info)

    # Include newly created target labels in tmp-context-products.json, this is to
    # include unregistered context in validation during development.
    create_tmp_context_json(proposal_id, data_dict, new_target_context_list)

def create_tmp_context_json(proposal_id, data_dict, targ_list=[]):
    """Create tmp-context-products.json to store additional context product information
    used for validation. It will be passed to --add-context-products parameter in the
    validation, for development purpose only.

    Inputs:
        proposal_id:    a proposal id.
        data_dict:      a data dictionary used to create the label.
    """
    # TODO: figure out the version id for all context data
    vid = '1.0'
    context_data = [
        f"urn:nasa:pds:context:investigation:individual.hst_{proposal_id:05}::{vid}",
        f"urn:nasa:pds:context:instrument_host:spacecraft.hst::{vid}"
    ]

    for inst in data_dict['inst_id_li']:
        context_data.append(f"urn:nasa:pds:context:instrument:hst.{inst.lower()}::{vid}")

    json_data = defaultdict(list)
    for data in context_data:
        new_context = {
            "name": [],
            "type": [],
            "lidvid": data,
        }
        json_data["Product_Context"].append(new_context)

    # Include newly added target labels
    for targ in targ_list:
        targ_name = targ['formatted_name']
        targ_type = targ['formatted_type']
        targ_lidvid = f"{targ['lid']}::{1.1}"
        new_context = {
            "name": [targ_name],
            "type": [targ_type],
            "lidvid": targ_lidvid,
        }
        json_data["Product_Context"].append(new_context)
    with open("tmp-context-products.json", "w") as tmp_json:
        json.dump(json_data, tmp_json)
