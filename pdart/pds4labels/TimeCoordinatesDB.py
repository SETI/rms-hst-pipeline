"""
Functionality to build an ``<Time_Coordinates />`` XML element using a
SQLite database.
"""
import pdart.add_pds_tools
import julian

from pdart.pds4labels.TimeCoordinatesXml import *
from pdart.rules.Combinators import *

from pdart.xml.Templates import NodeBuilder  # for mypy


def _db_get_start_stop_times_from_headers(headers):
    # type: (List[Dict[str, Any]]) -> Dict[str, str]
    date_obs = headers[0]['DATE-OBS']
    time_obs = headers[0]['TIME-OBS']
    exptime = headers[0]['EXPTIME']

    start_date_time = '%sT%sZ' % (date_obs, time_obs)
    stop_date_time = julian.tai_from_iso(start_date_time) + exptime
    stop_date_time = julian.iso_from_tai(stop_date_time,
                                         suffix='Z')

    return {'start_date_time': start_date_time,
            'stop_date_time': stop_date_time}


_db_get_start_stop_times = multiple_implementations(
    '_db_get_start_stop_times',
    _db_get_start_stop_times_from_headers,
    get_placeholder_start_stop_times)
# type: Callable[[List[Dict[str, Any]]], Dict[str, str]]


def get_db_time_coordinates(headers):
    # type: (List[Dict[str, Any]]) -> NodeBuilder
    """
    Create and return a ``<Time_Coordinates />`` XML element for the
    product.
    """
    return time_coordinates(_db_get_start_stop_times(headers))
