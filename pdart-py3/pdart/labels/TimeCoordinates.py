"""
Functionality to build an ``<Time_Coordinates />`` XML element using a
SQLite database.
"""
from typing import Any, Dict, List

import julian

from pdart.labels.TimeCoordinatesXml import time_coordinates
from pdart.xml.Templates import NodeBuilder


def _get_start_stop_times_from_cards(
    fits_product_lidvid: str, card_dicts: List[Dict[str, Any]]
) -> Dict[str, str]:
    date_obs = card_dicts[0]["DATE-OBS"]
    time_obs = card_dicts[0]["TIME-OBS"]
    exptime = float(card_dicts[0]["EXPTIME"])

    start_date_time = f"{date_obs}T{time_obs}Z"
    stop_date_time_float = julian.tai_from_iso(start_date_time) + exptime
    stop_date_time = julian.iso_from_tai(stop_date_time_float, suffix="Z")

    return {"start_date_time": start_date_time, "stop_date_time": stop_date_time}


def get_time_coordinates(
    fits_product_lidvid: str, card_dicts: List[Dict[str, Any]]
) -> NodeBuilder:
    """
    Create and return a ``<Time_Coordinates />`` XML element for the
    product.
    """
    return time_coordinates(
        _get_start_stop_times_from_cards(fits_product_lidvid, card_dicts)
    )
