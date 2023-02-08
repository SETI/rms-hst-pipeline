##########################################################################################
# hst_helper/general_utils.py
##########################################################################################
import csv
import datetime
import os
import pdslogger

from xmltemplate import XmlTemplate


def create_xml_label(template_path, label_path, data_dict, logger):
    """Create xml label with given template path, label path, and data dictionary.

    Inputs:
        template_path:    the path of the label template.
        label_path:       the path of the label to be created.
        logger:           pdslogger to use; None for default EasyLogger.
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
        csv_path:    the path of the csv file.
        data:        a list that contains row data to be written into the csv file. Each
                     row data in the list is a list of column values for the row.
        logger:      pdslogger to use; None for default EasyLogger.
    """
    logger = logger or pdslogger.EasyLogger()
    logger.info(f'Create csv: {csv_path}.')


    # open the file in the write mode
    with open(csv_path, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write rows to the csv file
        writer.writerows(data)
