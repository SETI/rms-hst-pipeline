##########################################################################################
# query_mast/utils.py
##########################################################################################

import julian

from product_labels.suffix_info import (ACCEPTED_SUFFIXES,
                                        ACCEPTED_LETTER_CODES, 
                                        INSTRUMENT_FROM_LETTER_CODE)

def ymd_tuple_to_mjd(ymd):
    """Return Modified Julian Date.
    Input:
        ymd:    a tuple of year, month, and day.
    """
    y, m, d = ymd
    days = julian.day_from_ymd(y, m, d)
    return julian.mjd_from_day(days)

def filter_table(row_predicate, table):
    """Return a copy of the filtered table object based on the return of row_predicate.
    Input:
        row_predicate:    a function with the condition used to filter the table.
        table:            target table to be filtered.
    """
    to_delete = [n for (n, row) in enumerate(table) if not row_predicate(row)]
    copy = table.copy()
    copy.remove_rows(to_delete)
    return copy

def is_accepted_instrument_letter_code(row):
    """Check if a product row has accepted letter code in the first letter of the 
    product filename.
    Input:
        row:    an observation table row.
    """
    return row["obs_id"][0].lower() in ACCEPTED_LETTER_CODES

def is_accepted_instrument_suffix(row):
    """Check if a product row has accepted suffex in the productSubGroupDescription field
    of the table.
    Input:
        row:    an observation table row.
    """
    suffix = get_suffix(row)
    instrument_id = get_instrument_id(row)
    # For files like n4wl03fxq_raw.jpg with "--" will raise an error
    # return is_accepted(suffix, instrument_id) 
    return suffix in ACCEPTED_SUFFIXES[instrument_id]

def get_instrument_id(row):
    """Return the instrument id for a given product row.
    Input:
        row:    an observation table row.
    """
    return INSTRUMENT_FROM_LETTER_CODE[row["obs_id"][0].lower()]

def get_suffix(row):
    """Return the product file suffix for a given product row.
    Input:
        row:    an observation table row.
    """
    return str(row["productSubGroupDescription"]).lower()