"""
Functionality to build an ``<Time_Coordinates />`` XML element using a
:class:`~pdart.reductions.Reduction.Reduction`.
"""
import pdart.add_pds_tools
import julian

from pdart.pds4labels.TimeCoordinatesXml import *
from pdart.reductions.Reduction import *
from pdart.rules.Combinators import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    # TODO mypy stubs for pyfits
    _HeaderUnit = Any


def _get_start_stop_times_from_header_unit(product_id, header_unit):
    # type: (unicode, _HeaderUnit) -> TemplateDict
    date_obs = header_unit['DATE-OBS']
    time_obs = header_unit['TIME-OBS']
    exptime = header_unit['EXPTIME']
    start_date_time = '%sT%sZ' % (date_obs, time_obs)
    stop_date_time = julian.tai_from_iso(start_date_time) + exptime
    stop_date_time = julian.iso_from_tai(stop_date_time,
                                         suffix='Z')
    return {'start_date_time': start_date_time,
            'stop_date_time': stop_date_time}


_get_start_stop_times = multiple_implementations(
    '_get_start_stop_times',
    _get_start_stop_times_from_header_unit,
    get_placeholder_start_stop_times)
# type: Callable[[unicode, _HeaderUnit], TemplateDict]


class TimeCoordinatesReduction(Reduction):
    """Reduce a product to a ``<Time_Coordinates />`` XML template."""
    def reduce_fits_file(self, file, get_reduced_hdus):
        get_start_stop_times = multiple_implementations(
            'get_start_stop_times',
            lambda: get_reduced_hdus()[0],
            lambda: get_placeholder_start_stop_times(
               'TimeCoordinatesReduction.reduce_fits_file', None))
        # type: Callable[[], TemplateDict]

        res = time_coordinates(get_start_stop_times())
        # type: NodeBuilder
        return res

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # returns dict or None
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        # returns dict or None
        if n == 0:
            return _get_start_stop_times(None, header_unit)
        else:
            pass
