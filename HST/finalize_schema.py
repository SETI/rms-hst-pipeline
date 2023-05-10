##########################################################################################
# finalize_schema.py
#
# Create schema directory, and schema csv & xml.
##########################################################################################

import pdslogger

from hst_helper import (PDS4_LIDVID,
                        HST_LIDVID,
                        DISP_LIDVID)
from hst_helper.fs_utils import (create_col_dir_in_bundle,
                                 get_formatted_proposal_id)
from hst_helper.general_utils import (create_collection_label,
                                      create_csv,
                                      get_mod_history_from_label)

CSV_FILENAME = 'collection_schema.csv'
COL_SCH_LABEL = 'collection_schema.xml'
COL_SCH_LABEL_TEMPLATE = 'SCHEMA_COLLECTION_LABEL.xml'

def label_hst_schema_directory(proposal_id, data_dict, logger=None, testing=False):
    """With a given proposal id, create schema directory in the final bundle. Return the
    path of the schema collectione label. These are the actions performed:

    1. Create schema directory.
    2. Create schema csv.
    3. create schema xml label.

    Inputs:
        proposal_id    a proposal id.
        data_dict      a data dictionary used to create the label.
        logger         pdslogger to use; None for default EasyLogger.
        testing        the flag used to determine if we are calling the function for
                       testing purpose with the test directory.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Label hst schema directory with proposal id: {proposal_id}')
    try:
        proposal_id = int(proposal_id)
    except ValueError:
        logger.exception(ValueError)
        raise ValueError(f'Proposal id: {proposal_id} is not valid.')

    # Create schema directory
    logger.info(f'Create schema directory for proposal id: {proposal_id}.')
    _, schema_dir = create_col_dir_in_bundle(proposal_id, 'schema', testing)

    # Create schema csv
    collection_schema_csv = f'{schema_dir}/{CSV_FILENAME}'
    collection_schema_data = [PDS4_LIDVID.split(','),
                              HST_LIDVID.split(','),
                              DISP_LIDVID.split(',')]
    create_schema_collection_csv(collection_schema_csv, collection_schema_data, logger)

    # TODO: Update the intelligence to determine the version and inventory nums later
    # Number of schema inventories: PDS4_LIDVID, HST_LIDVID, ISP_LIDVID
    records_num = 3
    # Get the mod history for schema collection label if it's already existed.
    version_id = (1, 0)
    col_sch_label_path = f'{schema_dir}/{COL_SCH_LABEL}'
    mod_history = get_mod_history_from_label(col_sch_label_path, version_id)

    sch_data_dict = {
        'collection_name': 'schema',
        'version_id': version_id,
        'csv_filename': CSV_FILENAME,
        'records_num': records_num,
        'mod_history': mod_history,
    }
    sch_data_dict.update(data_dict)

    # Create schema collection label
    return create_collection_label(proposal_id, 'schema', sch_data_dict,
                                   COL_SCH_LABEL, COL_SCH_LABEL_TEMPLATE,
                                   logger, testing)

def create_schema_collection_csv(csv_path, row_data, logger=None):
    """With a given proposal id, create schema collection csv in the final bundle.

    Inputs:
        csv_path    the path of the csv file.
        row_data    a list of row data to be written in the csv file. Each item of the
                    list is a list of column values for the row.
        logger      pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info('Create schema collection csv')

    create_csv(csv_path, row_data, logger)
