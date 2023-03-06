##########################################################################################
# finalize_schema/__init__.py
##########################################################################################
import os
import pdslogger

from hst_helper import (PDS4_LIDVID,
                        HST_LIDVID,
                        DISP_LIDVID)
from hst_helper.fs_utils import (get_formatted_proposal_id,
                                 get_program_dir_path)
from hst_helper.general_utils import (create_collection_label,
                                      create_csv,
                                      get_mod_history_from_label)

CSV_FILENAME = 'collection_schema.csv'
COL_SCH_LABEL = 'collection_schema.xml'
COL_SCH_LABEL_TEMPLATE = 'SCHEMA_COLLECTION_LABEL.xml'

def label_hst_schema_directory(proposal_id, data_dict, logger):
    """With a given proposal id, create schema directory in the final bundle.
    1. Create schema directory.
    2. Create schema csv.
    3. create schema xml label.


    Inputs:
        proposal_id:    a proposal id.
        data_dict:      a data dictionary used to create the label.
        logger:         pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst schema directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    formatted_proposal_id = get_formatted_proposal_id(proposal_id)

    # Create schema directory
    logger.info(f'Create schema directory for proposal id: {proposal_id}.')
    bundles_dir = get_program_dir_path(proposal_id, None, root_dir='bundles')
    schema_dir = bundles_dir + '/schema'
    os.makedirs(schema_dir, exist_ok=True)

    # Create schema csv
    collection_schema_csv = schema_dir + f'/{CSV_FILENAME}'
    collection_schema_data = [PDS4_LIDVID.split(','),
                              HST_LIDVID.split(','),
                              DISP_LIDVID.split(',')]
    create_schema_collection_csv(collection_schema_csv, collection_schema_data, logger)

    # TODO: Update the intelligence to determine the version and inventory nums later
    # Number of schema inventories: PDS4_LIDVID, HST_LIDVID, ISP_LIDVID
    records_num = 3
    # Get the mod history for schema collection label if it's already existed.
    version_id = (1, 0)
    col_sch_label_path = bundles_dir + f'/schema/{COL_SCH_LABEL}'
    mod_history = get_mod_history_from_label(col_sch_label_path, version_id)

    sch_data_dict = {
        'collection_name': 'schema',
        'version_id': version_id,
        'csv_filename': CSV_FILENAME,
        'records_num': records_num,
        'mod_history': mod_history,
    }
    sch_data_dict = {**sch_data_dict, **data_dict}

    # Create schema collection label
    create_collection_label(proposal_id, 'schema', sch_data_dict,
                            COL_SCH_LABEL, COL_SCH_LABEL_TEMPLATE, logger)

def create_schema_collection_csv(csv_path, row_data, logger):
    """With a given proposal id, create schema collection csv in the final bundle.

    Inputs:
        csv_path:   the path of the csv file.
        row_data:   a list of row data to be written in the csv file. Each item of the
                    list is a list of column values for the row.
        logger:     pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info('Create schema collection csv')

    create_csv(csv_path, row_data, logger)
