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

_ACCEPTED_SUFFIXES = [
    'C0F', 'C1F', 'C3F', 'CRJ', 'D0F', 'DRZ', 'FLT', 'RAW'
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
    return row['productSubGroupDescription'].upper() in _ACCEPTED_SUFFIXES


class MastSlice(object):
    """
    A slice in time of the MAST database.

    Column names for observations are: 'calib_level', 'dataRights',
    'dataURL', 'dataproduct_type', 'em_max', 'em_min', 'filters',
    'instrument_name', 'intentType', 'jpegURL', 'mtFlag', 'objID',
    'obs_collection', 'obs_id', 'obs_title', 'obsid', 'project',
    'proposal_id', 'proposal_pi', 'proposal_type', 's_dec', 's_ra',
    's_region', 'srcDen', 't_exptime', 't_max', 't_min',
    't_obs_release', 'target_classification', 'target_name',
    'wavelength_region'

    Column names for products are: 'dataURI', 'dataproduct_type',
    'description', 'obsID', 'obs_collection', 'obs_id',
    'parent_obsid', 'productDocumentationURL', 'productFilename',
    'productGroupDescription', 'productSubGroupDescription',
    'productType', 'project', 'proposal_id', 'prvversion', 'size',
    'type'
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
