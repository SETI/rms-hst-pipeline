"""
Functionality to build an ``<Time_Coordinates />`` XML element using a
SQLite database.
"""
from typing import Any, Dict, List

import julian

from pdart.labels.Lookup import Lookup
from pdart.labels.TimeCoordinatesXml import time_coordinates
from pdart.xml.Templates import NodeBuilder


def get_start_stop_times(lookup: Lookup) -> Dict[str, str]:
    try:
        return get_start_stop_times_old(lookup)
    except KeyError:
        # TODO return dummy values
        return {
            "start_date_time": "2001-01-01T00:00:28Z",
            "stop_date_time": "2001-01-01T00:00:30Z",
            "exposure_duration": "2.0",
        }


def get_start_stop_times_old(lookup: Lookup) -> Dict[str, str]:
    # TODO Remove this after get_start_stop_times() is fixed.
    date_obs, time_obs, exptime = lookup.keys(["DATE-OBS", "TIME-OBS", "EXPTIME"])

    start_date_time = f"{date_obs}T{time_obs}Z"
    stop_date_time_float = julian.tai_from_iso(start_date_time) + float(exptime)
    stop_date_time = julian.iso_from_tai(stop_date_time_float, suffix="Z")

    return {
        "start_date_time": start_date_time,
        "stop_date_time": stop_date_time,
        "exposure_duration": exptime,
    }


def get_time_coordinates(start_stop_times: Dict[str, str]) -> NodeBuilder:
    """
    Create and return a ``<Time_Coordinates />`` XML element for the
    product.
    """
    return time_coordinates(start_stop_times)
