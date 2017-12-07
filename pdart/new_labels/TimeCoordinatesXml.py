"""
Templates to create a ``<Time_Coordinates />`` XML node for product
labels.
"""
from typing import TYPE_CHECKING

from pdart.new_labels.Placeholders import placeholder_time
from pdart.xml.Templates import *

if TYPE_CHECKING:
    from typing import Any, Dict, List

time_coordinates = interpret_template("""<Time_Coordinates>
      <start_date_time><NODE name="start_date_time"/></start_date_time>
      <stop_date_time><NODE name="stop_date_time"/></stop_date_time>
    </Time_Coordinates>""")
# type: NodeBuilderTemplate
"""
An interpreted node builder template to create an ``<Time_Coordinates />``
XML element.
"""


def _remove_trailing_decimal(str):
    # type: (str) -> str
    """
    Given a string, remove any trailing zeros and then any trailing
    decimal point and return it.
    """
    # remove any trailing zeros
    while str[-1] == '0':
        str = str[:-1]
    # remove any trailing decimal point
    if str[-1] == '.':
        str = str[:-1]
    return str


def get_placeholder_start_stop_times(fits_product_lidvid, headers):
    # type: (unicode, List[Dict[str, Any]]) -> Dict[unicode, unicode]
    """
    Return a placeholder ``<Time_Coordinates />`` XML element.
    """
    start_date_time = placeholder_time(fits_product_lidvid, 'Time_Coordinates')
    stop_date_time = start_date_time
    return {'start_date_time': start_date_time,
            'stop_date_time': stop_date_time}
