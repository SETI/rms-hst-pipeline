"""
Functionality to build an ``<Time_Coordinates />`` XML element using a
SQLite database.
"""
import pdart.add_pds_tools
import julian
from typing import TYPE_CHECKING

from pdart.new_labels.TimeCoordinatesXml \
    import get_placeholder_start_stop_times, time_coordinates
from pdart.rules.Combinators import multiple_implementations

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List
    from pdart.xml.Templates import NodeBuilder


def _to_string(in_str):
    # type: (unicode) -> str
    return in_str.encode('ascii', 'ignore')


def _get_start_stop_times_from_cards(fits_product_lidvid, card_dicts):
    # type: (unicode, List[Dict[str, Any]]) -> Dict[str, str]
    date_obs = _to_string(card_dicts[0]['DATE-OBS'])
    time_obs = _to_string(card_dicts[0]['TIME-OBS'])
    exptime = float(card_dicts[0]['EXPTIME'])

    start_date_time = '%sT%sZ' % (date_obs, time_obs)
    stop_date_time = julian.tai_from_iso(start_date_time) + exptime
    stop_date_time = julian.iso_from_tai(stop_date_time,
                                         suffix='Z')

    return {'start_date_time': start_date_time,
            'stop_date_time': stop_date_time}


_get_start_stop_times = multiple_implementations(
    '_get_start_stop_times',
    _get_start_stop_times_from_cards,
    get_placeholder_start_stop_times)
# type: Callable[[unicode, List[Dict[str, Any]]], Dict[str, str]]


def get_time_coordinates(fits_product_lidvid, card_dicts):
    # type: (unicode, List[Dict[str, Any]]) -> NodeBuilder
    """
    Create and return a ``<Time_Coordinates />`` XML element for the
    product.
    """
    return time_coordinates(_get_start_stop_times(fits_product_lidvid,
                                                  card_dicts))
