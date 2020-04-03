"""
Templates to create a ``<Time_Coordinates />`` XML node for product
labels.
"""

from pdart.labels.Placeholders import placeholder_time
from pdart.xml.Templates import interpret_template


from typing import Any, Dict, List
from pdart.xml.Templates import NodeBuilderTemplate

time_coordinates: NodeBuilderTemplate = interpret_template(
    """<Time_Coordinates>
      <start_date_time><NODE name="start_date_time"/></start_date_time>
      <stop_date_time><NODE name="stop_date_time"/></stop_date_time>
    </Time_Coordinates>"""
)
"""
An interpreted node builder template to create an ``<Time_Coordinates />``
XML element.
"""


def _remove_trailing_decimal(num_str: str) -> str:
    """
    Given a num_string, remove any trailing zeros and then any trailing
    decimal point and return it.
    """
    # remove any trailing zeros
    while num_str[-1] == "0":
        num_str = num_str[:-1]
    # remove any trailing decimal point
    if num_str[-1] == ".":
        num_str = num_str[:-1]
    return num_str


def get_placeholder_start_stop_times(
    fits_product_lidvid: str, headers: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Return a placeholder ``<Time_Coordinates />`` XML element.
    """
    start_date_time = placeholder_time(fits_product_lidvid, "Time_Coordinates")
    stop_date_time = start_date_time
    return {"start_date_time": start_date_time, "stop_date_time": stop_date_time}
