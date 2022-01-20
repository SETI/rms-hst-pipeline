from typing import Iterable, List, Optional, Tuple, cast

from astropy.table import Table
from astropy.table.row import Row
from astroquery.mast import Observations

from pdart.astroquery.Utils import (
    filter_table,
    get_table_with_retries,
    ymd_tuple_to_mjd,
)

from pdart.pipeline.SuffixInfo import (  # type: ignore
    IDENTIFICATION_SUFFIXES,
    TARGET_IDENTIFICATION_SUFFIXES,
    ACCEPTED_INSTRUMENTS,
    INSTRUMENTS_INFO,
    INTRUMENT_SELECTED_SUFFIXES,
)

_YMD = Tuple[int, int, int]

"""
We currently only handle products from a limited set of
instruments.  These are the first letters of their 'obs_id's.
"""


def _instrument_key(id: str) -> str:
    """
    Return the first letter of the obs_id, which tells which
    instrument made the observation.
    """
    return id[0].upper()


def _is_accepted_instrument_product_row(row: Row) -> bool:
    """
    We currently only handle products from a limited set of
    instruments.
    """
    return _instrument_key(row["obs_id"]) in ACCEPTED_INSTRUMENTS


def _is_accepted_product_type_product_row(row: Row) -> bool:
    """
    We currently only handle products from a limited set of
    instruments.
    """
    desc = str(row["productSubGroupDescription"])
    instrument = INSTRUMENTS_INFO[_instrument_key(row["obs_id"]).lower()]
    selected_suffixes = INTRUMENT_SELECTED_SUFFIXES[instrument]
    return desc.upper() in selected_suffixes


def _is_selected_accepted_product_type_product_row(row: Row) -> bool:
    """
    We currently only handle products from a limited set of
    instruments with selected suffixes.
    """
    desc = str(row["productSubGroupDescription"])
    return desc.lower() in TARGET_IDENTIFICATION_SUFFIXES


class MastSlice(object):
    """
    A slice in time of the MAST database.

    Look in test_Astroquery.py for a current list of the columns
    returned for observations and products.
    """

    def __init__(
        self, start_date: _YMD, end_date: _YMD, proposal_id: Optional[int] = None
    ) -> None:
        """
        Given a start and an end date expressed as ymd triples,
        download a slice of the MAST database and express it as an
        object.
        """
        self.start_date = ymd_tuple_to_mjd(start_date)
        self.end_date = ymd_tuple_to_mjd(end_date)

        def mast_call() -> Table:
            if proposal_id is not None:
                return Observations.query_criteria(
                    dataproduct_type=["image"],
                    dataRights="PUBLIC",
                    obs_collection=["HST"],
                    proposal_id=str(proposal_id),
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True,
                )
            else:
                return Observations.query_criteria(
                    dataproduct_type=["image"],
                    dataRights="PUBLIC",
                    obs_collection=["HST"],
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True,
                )

        self.observations_table = get_table_with_retries(mast_call, 1)
        self.proposal_ids: Optional[List[int]] = None

    def __str__(self) -> str:
        return f"MastSlice(julian day [{self.start_date}, {self.end_date}])"

    def get_proposal_ids(self) -> List[int]:
        if self.proposal_ids is None:
            result = [int(id) for id in self.observations_table["proposal_id"]]
            self.proposal_ids = sorted(list(set(result)))
        return self.proposal_ids

    def get_products(self, proposal_id: int, selected_suffixes: bool = False) -> Table:
        def proposal_id_matches(row: Row) -> bool:
            return int(row["proposal_id"]) == proposal_id

        proposal_table = filter_table(proposal_id_matches, self.observations_table)

        result = Observations.get_product_list(proposal_table)
        result = filter_table(_is_accepted_instrument_product_row, result)
        if selected_suffixes:
            result = filter_table(
                _is_selected_accepted_product_type_product_row, result
            )
        else:
            result = filter_table(_is_accepted_product_type_product_row, result)
        return result

    def to_product_set(
        self, proposal_id: int, selected_suffixes: bool = False
    ) -> "ProductSet":
        return ProductSet(self.get_products(proposal_id, selected_suffixes))

    def download_products(self, products_table: Table, download_dir: str) -> None:
        if len(products_table) > 0:
            Observations.download_products(
                products_table, mrp_only=False, download_dir=download_dir
            )

    def download_products_by_id(self, proposal_id: int, download_dir: str) -> None:
        products_table = self.get_products(proposal_id)
        self.download_products(products_table, download_dir=download_dir)


# MastSlice with customized query criteria
class CustomizedQueryMastSlice(MastSlice):
    def __init__(
        self, start_date: _YMD, end_date: _YMD, proposal_id: Optional[int] = None
    ) -> None:
        super().__init__(start_date, end_date, proposal_id)

        # Don't restrict product type to image
        def mast_call() -> Table:
            if proposal_id is not None:
                return Observations.query_criteria(
                    dataRights="PUBLIC",
                    obs_collection=["HST"],
                    proposal_id=str(proposal_id),
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True,
                )
            else:
                return Observations.query_criteria(
                    dataRights="PUBLIC",
                    obs_collection=["HST"],
                    t_obs_release=(self.start_date, self.end_date),
                    mtFlag=True,
                )

        self.observations_table = get_table_with_retries(mast_call, 1)
        self.proposal_ids: Optional[List[int]] = None

    # Don't restrict the file suffixes, get all files.
    def get_products(self, proposal_id: int, selected_suffixes: bool = False) -> Table:
        def proposal_id_matches(row: Row) -> bool:
            return int(row["proposal_id"]) == proposal_id

        proposal_table = filter_table(proposal_id_matches, self.observations_table)

        result = Observations.get_product_list(proposal_table)
        result = filter_table(_is_accepted_instrument_product_row, result)

        if selected_suffixes:
            result = filter_table(
                _is_selected_accepted_product_type_product_row, result
            )

        return result


def products_size(table: Table) -> int:
    return sum(cast(Iterable[int], table["size"]))


class ProductSet(object):
    """
    A downloadable collection of MAST products.
    """

    def __init__(self, table: Table) -> None:
        self.table = table

    def product_count(self) -> int:
        return len(self.table)

    def download_size(self) -> int:
        return sum(cast(Iterable[int], self.table["size"]))

    def download(self, download_dir: str) -> None:
        if len(self.table) > 0:
            Observations.download_products(self.table, download_dir=download_dir)
