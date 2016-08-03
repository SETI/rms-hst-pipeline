from pdart.exceptions.Combinators import *
from pdart.pds4labels.TimeCoordinatesXml import *


def _db_get_start_stop_times_from_headers(headers):
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


def get_db_time_coordinates(headers, conn, lid):
    return time_coordinates(_db_get_start_stop_times(headers))
