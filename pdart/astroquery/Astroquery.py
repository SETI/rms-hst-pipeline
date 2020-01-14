from astroquery.mast import Observations
import numpy
from typing import TYPE_CHECKING

from pdart.astroquery.Utils import filter_table, get_table_with_retries, \
    ymd_tuple_to_mjd

if TYPE_CHECKING:
    from astroquery.table import Table
    from typing import List, Optional, Tuple
    _YMD = Tuple[int, int, int]

_ACCEPTED_INSTRUMENTS = "IJU"  # type: str
"""
We currently only handle products from a limited set of
instruments.  These are the first letters of their 'obs_id's.
"""

_ACCEPTED_SUFFIXES = [
    'A1F', 'A2F', 'A3F', 'ASC', 'ASN', 'C0F', 'C1F', 'C2F', 'C3F', 'CAL',
    'CLB', 'CLF', 'CORRTAG', 'CQF', 'CRJ', 'D0F', 'DRC', 'DRZ', 'FLC', 'FLT',
    'FLTSUM', 'MOS', 'RAW', 'SHM', 'SX2', 'SXL', 'X1D', 'X1DSUM', 'X2D'
]  # type: List[str]
"""
For now, we limit the types of the products to those with these
suffixes.
"""


def _is_accepted_instrument_product_row(row):
    # type: (Table) -> bool
    """
    We currently only handle products from a limited set of
    instruments.
    """

    def instrument_key(id):
        """
        Return the first letter of the obs_id, which tells which
        instrument made the observation.
        """
        return id[0].upper()

    return instrument_key(row['obs_id']) in _ACCEPTED_INSTRUMENTS


def _is_accepted_product_type_product_row(row):
    # type: (Table) -> bool
    """
    We currently only handle products from a limited set of
    instruments.
    """
    desc = str(row['productSubGroupDescription'])
    return desc.upper() in _ACCEPTED_SUFFIXES


class MastSlice(object):
    """
    A slice in time of the MAST database.

    Look in test_Astroquery.py for a current list of the columns
    returned for observations and products.
    """

    def __init__(self, start_date, end_date, proposal_id=None):
        # type: (_YMD, _YMD, Optional[int]) -> None
        """
        Given a start and an end date expressed as ymd triples,
        download a slice of the MAST database and express it as an
        object.
        """
        self.start_date = ymd_tuple_to_mjd(start_date)
        self.end_date = ymd_tuple_to_mjd(end_date)

        def mast_call():
            # type: () -> Table
            if proposal_id is not None:
                return Observations.query_criteria(
                    dataproduct_type=['image'],
                    dataRights='PUBLIC',
                    obs_collection=['HST'],
                    proposal_id=str(proposal_id),
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True)
            else:
                return Observations.query_criteria(
                    dataproduct_type=['image'],
                    dataRights='PUBLIC',
                    obs_collection=['HST'],
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True)

        self.observations_table = get_table_with_retries(mast_call, 1)
        self.proposal_ids = None  # type: Optional[List[int]]

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

        result = Observations.get_product_list(proposal_table)
        result = filter_table(_is_accepted_instrument_product_row,
                              result)
        result = filter_table(_is_accepted_product_type_product_row, result)
        return result

    def to_product_set(self, proposal_id):
        # type: (int) -> ProductSet
        return ProductSet(self.get_products(proposal_id))

    def download_products(self, products_table, download_dir):
        # type: (Table, unicode) -> None
        if len(products_table) > 0:
            Observations.download_products(products_table,
                                           mrp_only=False,
                                           download_dir=download_dir)

    def download_products_by_id(self, proposal_id, download_dir):
        # type: (int, unicode) -> None
        products_table = self.get_products(proposal_id)
        self.download_products(products_table,
                               download_dir=download_dir)


def products_size(table):
    return sum(table['size'])


class ProductSet(object):
    """
    A downloadable collection of MAST products.
    """

    def __init__(self, table):
        # type: (Table) -> None
        self.table = table

    def product_count(self):
        # type: () -> int
        return len(self.table)

    def download_size(self):
        # type: () -> int
        return sum(self.table['size'])

    def download(self, download_dir):
        # type: (unicode) -> None
        if len(self.table) > 0:
            Observations.download_products(self.table,
                                           download_dir=download_dir)
