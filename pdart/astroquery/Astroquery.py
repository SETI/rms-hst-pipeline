from astroquery.mast import Observations
from typing import TYPE_CHECKING

from pdart.astroquery.Utils import filter_table, get_table_with_retries, \
    ymd_tuple_to_mjd

if TYPE_CHECKING:
    from astroquery.table import Table
    from typing import Tuple

_ACCEPTED_INSTRUMENTS = "IJU"  # type: str
"""
We currently only handle products from a limited set of
instruments.  These are the first letters of their 'obs_id's.
"""


def _is_accepted_instrument_product_row(row):
    # type: (Table) -> bool
    """
    We currently only handle products from a limited set of
    instruments.
    """
    return row['obs_id'][0].upper() in _ACCEPTED_INSTRUMENTS


class MastSlice(object):
    """
    A slice in time of the MAST database.
    """

    def __init__(self, start_date, end_date):
        # type: (Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Given a start and an end date expressed as ymd triples,
        download a slice of the MAST database and express it as an
        object.
        """
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
        self.proposal_ids = None  # type: List[int]

    def __str__(self):
        return 'MastSlice(julian day [%f, %f])' % (
            self.start_date, self.end_date)

    def get_proposal_ids(self):
        # type: () -> List[int]
        if self.proposal_ids is None:
            result = [int(id) for id in self.observations_table['proposal_id']]
            self.proposal_ids = sorted(list(set(result)))
        return self.proposal_ids

    def get_products(self, proposal_id):
        # type: (int) -> Table
        def proposal_id_matches(row):
            return int(row['proposal_id']) == proposal_id

        proposal_table = filter_table(proposal_id_matches,
                                      self.observations_table)

        products_table = Observations.get_product_list(proposal_table)
        result = filter_table(_is_accepted_instrument_product_row,
                              products_table)
        return result
