from astroquery.mast import Observations
from typing import TYPE_CHECKING

from pdart.astroquery.Utils import get_table_with_retries, ymd_tuple_to_mjd

if TYPE_CHECKING:
    from astroquery.table import Table
    from typing import Tuple


class MastSlice(object):
    def __init__(self, start_date, end_date):
        # type: (Tuple[int, int, int], Tuple[int, int, int]) -> None
        self.start_date = ymd_tuple_to_mjd(start_date)
        self.end_date = ymd_tuple_to_mjd(end_date)

        def mast_call():
            # type: () -> Table
            return Observations.query_criteria(
                dataproduct_type=['image'],
                dataRights='PUBLIC',
                obs_collection=['HST'],
                t_obs_release=(self.start_date, self.end_date),
                mtFlag=True)

        self.observations_table = get_table_with_retries(mast_call, 1)

    def __str__(self):
        return 'MastSlice(julian day [%f, %f])' % (
            self.start_date, self.end_date)
