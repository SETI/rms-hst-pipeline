import pdart.add_pds_tools
import julian

from pdart.exceptions.Combinators import *
from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


# For product labels: produces the Time_Coordinates element.

time_coordinates = interpret_template("""<Time_Coordinates>
      <start_date_time><NODE name="start_date_time"/></start_date_time>
      <stop_date_time><NODE name="stop_date_time"/></stop_date_time>
    </Time_Coordinates>""")


def _remove_trailing_decimal(str):
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


def _get_time_coordinates_from_header_unit(header_unit):
    date_obs = header_unit['DATE-OBS']
    time_obs = header_unit['TIME-OBS']
    exptime = header_unit['EXPTIME']
    start_date_time = '%sT%sZ' % (date_obs, time_obs)
    stop_date_time = julian.tai_from_iso(start_date_time) + exptime
    stop_date_time = julian.iso_from_tai(stop_date_time,
                                         suffix='Z')
    return time_coordinates({'start_date_time': start_date_time,
                             'stop_date_time': stop_date_time})


def _get_placeholder_time_coordinates(header_unit):
    start_date_time = '2000-01-02Z'
    stop_date_time = '2000-01-02Z'
    return time_coordinates({'start_date_time': start_date_time,
                             'stop_date_time': stop_date_time})


_get_time_coordinates = multiple_implementations(
    '_get_time_coordinates',
    _get_time_coordinates_from_header_unit,
    _get_placeholder_time_coordinates)


class TimeCoordinatesLabelReduction(Reduction):
    """Reduce a product to an XML Time_Coordinates node template."""
    def reduce_fits_file(self, file, get_reduced_hdus):
        # returns Doc -> Node
        reduced_hdus = get_reduced_hdus()
        return reduced_hdus[0]

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # returns (Doc -> Node) or None
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        # returns (Doc -> Node) or None
        if n == 0:
            return _get_time_coordinates(header_unit)
        else:
            pass
