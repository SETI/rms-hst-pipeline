##########################################################################################
# hst_helper/general_utils.py
#
# This file contains general helper functions like creating csv & xml based on the passed
# in data, getting data info from xml label, getting instrument ids for a given proposal
# id, getting citation info, and etc.
##########################################################################################

import csv
import os
import pdslogger

from . import (CITATION_INFO_DICT,
               DOCUMENT_EXT_FOR_CITATION_INFO,
               INST_ID_DICT,
               INST_PARAMS_DICT,
               PRIMARY_RES_DICT,
               PROGRAM_INFO_FILE,
               RECORDS_DICT,
               TARG_ID_DICT,
               TIME_DICT)
from .fs_utils import (get_deliverable_path,
                       get_formatted_proposal_id,
                       get_program_dir_path,
                       get_instrument_id_from_fname)
from citations import Citation_Information
from product_labels.xml_support import (get_instrument_params,
                                        get_modification_history,
                                        get_primary_result_summary,
                                        get_target_identifications,
                                        get_time_coordinates)
from xmltemplate import XmlTemplate

def create_collection_label(
    proposal_id, collection_name, data_dict,
    label_name, template_name, logger=None, testing=False
):
    """With a given proposal id, create collection label in the final bundle.

    Inputs:
        proposal_id        a proposal id.
        collection_name    collection name in the bundles.
        data_dict          data dictonary to fill in the label template.
        label_name         the name of the collection label
        template_name      the name of the template being used.
        logger             pdslogger to use; None for default EasyLogger.
        target_dir         the target dir used to obtain the roll up info.
        testing            the flag used to determine if we are calling the function for
                           testing purpose with the test directory.

    Returns:    the path of the newly created collection label.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create collection csv with proposal id: {proposal_id}')

    # Create collection label
    logger.info(f'Create label for collection using templates/{template_name}.')
    # Collection label template path
    col_dir = os.path.dirname(os.path.abspath(__file__))
    col_template = f'{col_dir}/../templates/{template_name}'
    # Collection label path
    deliverable_path = get_deliverable_path(proposal_id, testing)
    if collection_name == 'bundle':
        col_label_path = f'{deliverable_path}/{label_name}'
    else:
        col_label_path = f'{deliverable_path}/{collection_name}/{label_name}'

    create_xml_label(col_template, col_label_path, data_dict, logger)

    return col_label_path

def create_xml_label(template_path, label_path, data_dict, logger):
    """Create xml label with given template path, label path, and data dictionary.

    Inputs:
        template_path    the path of the label template.
        label_path       the path of the label to be created.
        logger           pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create label using template from: {template_path}.')
    TEMPLATE = XmlTemplate(template_path)
    XmlTemplate.set_logger(logger)

    logger.info('Insert data to the label template.')
    TEMPLATE.write(data_dict, label_path)
    if TEMPLATE.ERROR_COUNT == 1:
        logger.error('1 error encountered', label_path)
    elif TEMPLATE.ERROR_COUNT > 1:
        logger.error(f'{TEMPLATE.ERROR_COUNT} errors encountered', label_path)

def create_csv(csv_path, data, logger):
    """Create csv with given csv file path and data to be written into the csv file.

    Inputs:
        csv_path    the path of the csv file.
        data        a list that contains row data to be written into the csv file. Each
                    row data in the list is a list of column values for the row.
        logger      pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create csv: {csv_path}.')


    # open the file in the write mode
    with open(csv_path, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write rows to the csv file
        writer.writerows(data)

def get_citation_info(proposal_id, logger):
    """Search for proposal files & program info file stored at pipeline directory to
    obtain the citation info for a given proposal id.

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.

    Returns:    the Citation_Information object of the given proposal id.
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    if formatted_proposal_id in CITATION_INFO_DICT:
        return CITATION_INFO_DICT[formatted_proposal_id]
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Get citation info for: {proposal_id}.')

    pipeline_dir = get_program_dir_path(proposal_id, None, root_dir='pipeline')

    for file in os.listdir(pipeline_dir):
        _, _, ext = file.rpartition('.')
        # We don't have an implementation to create citation info from pdf.
        if ext in DOCUMENT_EXT_FOR_CITATION_INFO:
            file_path = f'{pipeline_dir}/{file}'
            if formatted_proposal_id not in CITATION_INFO_DICT:
                CITATION_INFO_DICT[formatted_proposal_id] = (
                    Citation_Information.create_from_file(file_path)
                )
            return CITATION_INFO_DICT[formatted_proposal_id]

def get_instrument_id_set(proposal_id, logger):
    """Walk through all downloaded files, store data in the INST_ID_DICT, and return
    a set of instrument ids for a given propsal id.

    Inputs:
        proposal_id    a proposal id.
        logger         pdslogger to use; None for default EasyLogger.

    Returns:    a set of instrument ids for the given proposal id.
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Get instrument id
    if formatted_proposal_id not in INST_ID_DICT:
        logger = logger or pdslogger.EasyLogger()
        logger.info(f'Get instrument ids for: {proposal_id}.')
        # Walk through all the downloaded files from MAST in staging directory
        files_dir = get_program_dir_path(proposal_id, None, root_dir='staging')
        for root, dirs, files in os.walk(files_dir):
            for file in files:
                inst_id = get_instrument_id_from_fname(file)
                formatted_proposal_id = str(proposal_id).zfill(5)
                if inst_id is not None:
                    INST_ID_DICT[formatted_proposal_id].add(inst_id)

    return INST_ID_DICT[formatted_proposal_id]

def get_mod_history_from_label(prev_label_path, current_version_id):
    """Compare current version id with the one in the exisiting (old) label. Return a
    list of modification history to be used in new label.

    Inputs:
        prev_label_path       the path of the exisiting xml label
        current_version_id    the current version id of the new bundle

    Returns:    a list of tuples (modification_date, version_id, description), one for
                each Modification_Detail.
    """
    mod_history = []
    if os.path.exists(prev_label_path):
        with open(prev_label_path) as f:
            xml_content = f.read()
            modification_history = get_modification_history(xml_content)
            old_version = modification_history[-1][1]
            if old_version != current_version_id:
                mod_history = modification_history

    return mod_history

def get_target_id_from_label(proposal_id, prev_label_path):
    """Get the target identification info from the exisitng label.

    Inputs:
        proposal_id        a proposal id.
        prev_label_path    the path of the exisiting xml label

    Returns:    a list of the target identification info tuples (name,
                alternate_designations, type, description, lid_reference)
        name                the preferred name;
        alt_designations    a list of strings indicating alternative names;
        body_type           "Asteroid", "Centaur", etc.;
        description         a list of strings, to be separated by newlines inside the
                            description attribute of the XML Target_Identification object;
        lid                 the LID of the object, omitting "urn:...:target:".
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    if formatted_proposal_id not in TARG_ID_DICT:
        if os.path.exists(prev_label_path):
            with open(prev_label_path) as f:
                xml_content = f.read()
                # use the old identification if available
                if xml_content:
                        target_ids = get_target_identifications(xml_content)
                        for targ in target_ids:
                            if targ not in TARG_ID_DICT[formatted_proposal_id]:
                                TARG_ID_DICT[formatted_proposal_id].append(targ)

    return TARG_ID_DICT[formatted_proposal_id]

def date_time_to_date(date_time):
    """
    Take in a date_time string and return a date string, for exampel:
    "2005-01-19T15:41:05Z" to "2005-01-19"

    Inputs:
        date_time    a date time string like "2005-01-19T15:41:05Z".

    Returns:    a date string.
    """
    try:
        idx = date_time.index('T')
    except:
        raise ValueError('Failed to convert from date_time to date')

    return date_time[:idx]

def get_collection_label_data(proposal_id, target_dir, logger):
    """Walk through the given target directory of a proposal id to get the collection
    label data used for label creation.

    Inputs:
        proposal_id    a proposal id.
        target_dir     the targeted labels directory to get the number of total files.

    Returns:    a dictionary containing target id, time, instrument params, primary
                results and the number of records. Here are the keys of the returned
                dictionary 'target', 'time', 'inst_params', 'primary_res', and 'records'.
    """
    formatted_proposal_id = get_formatted_proposal_id(proposal_id)
    _, _, collection_name = target_dir.rpartition('/')
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Get collection label data for: {proposal_id} {collection_name}')
    res = {}

    files_li = []
    min_start = None
    max_stop = None

    if formatted_proposal_id in TARG_ID_DICT:
        res['target'] = TARG_ID_DICT[formatted_proposal_id]
    if (formatted_proposal_id in TIME_DICT and
        collection_name in TIME_DICT[formatted_proposal_id]):
        res['time'] = TIME_DICT[formatted_proposal_id][collection_name]
    if (formatted_proposal_id in INST_PARAMS_DICT and
        collection_name in INST_PARAMS_DICT[formatted_proposal_id]):
        res['inst_params'] = INST_PARAMS_DICT[formatted_proposal_id][collection_name]
    if (formatted_proposal_id in PRIMARY_RES_DICT and
        collection_name in PRIMARY_RES_DICT[formatted_proposal_id]):
        res['primary_res'] = PRIMARY_RES_DICT[formatted_proposal_id][collection_name]
    if (formatted_proposal_id in RECORDS_DICT and
        collection_name in RECORDS_DICT[formatted_proposal_id]):
        res['records'] = RECORDS_DICT[formatted_proposal_id][collection_name]

    for root, _, files in os.walk(target_dir):
            for file in files:
                if not file.startswith('collection_') and file.endswith('.xml'):
                    file_path = os.path.join(root, file)
                    with open(file_path) as f:
                        xml_content = f.read()
                        # target identifications
                        if 'target' not in res:
                            target_ids = get_target_identifications(xml_content)
                            for targ in target_ids:
                                if targ not in TARG_ID_DICT[formatted_proposal_id]:
                                    TARG_ID_DICT[formatted_proposal_id].append(targ)
                        # roll up start/stop time
                        if 'time' not in res:
                            start, stop = get_time_coordinates(xml_content)
                            min_start = start if min_start is None else min(min_start,
                                                                            start)
                            max_stop = stop if max_stop is None else max(max_stop, stop)
                        # instrument params
                        if 'inst_params' not in res:
                            INST_PARAMS_DICT[formatted_proposal_id][collection_name]  = (
                                get_instrument_params(xml_content)
                            )
                            res['inst_params'] = (INST_PARAMS_DICT[formatted_proposal_id]
                                                                  [collection_name])
                        # primary results
                        if 'primary_res' not in res:
                            PRIMARY_RES_DICT[formatted_proposal_id][collection_name] = (
                                get_primary_result_summary(xml_content)
                            )
                            res['primary_res'] = (PRIMARY_RES_DICT[formatted_proposal_id]
                                                                  [collection_name])
                        # records
                        if file not in files_li:
                            files_li.append(file)

    if 'target' not in res:
        res['target'] = TARG_ID_DICT[formatted_proposal_id]
    if 'time' not in res:
        TIME_DICT[formatted_proposal_id][collection_name] = (min_start, max_stop)
        res['time'] = TIME_DICT[formatted_proposal_id][collection_name]
    if 'records' not in res:
        RECORDS_DICT[formatted_proposal_id][collection_name] = len(files_li)
        res['records'] = RECORDS_DICT[formatted_proposal_id][collection_name]

    return res

def get_clean_target_text(text: str) -> str:
    """Get the target text used in target label in PDS page.

    Inputs:
        text    a text of the target name or type.

    Returns:    a string with special characters replaced by '_' and '()' removed.
    """
    SPECIAL_CHARS = '!#$%^&*/ '
    REMOVED_CHARS = '()'
    for char in SPECIAL_CHARS:
        text = text.replace(char, '_')
    for char in REMOVED_CHARS:
        text = text.replace(char, '')
    return text
