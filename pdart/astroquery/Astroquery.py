from typing import TYPE_CHECKING
from astroquery.mast import Observations
from pdart.astroquery.Utils import mjd_range_to_now

if TYPE_CHECKING:
    from astropy.table import Table
    from typing import Any, Callable, Set


def get_hst_moving_images(last_update_mjd):
    # type: (float) -> Table
    return Observations.query_criteria(
        dataRights='PUBLIC',
        dataproduct_type=['image'],
        t_obs_release=mjd_range_to_now(last_update_mjd),
        obs_collection=['HST'],
        mtFlag=True)


def get_hst_by_proposal_id(prop_id):
    # type: (float) -> Table
    return Observations.query_criteria(
        proposal_id=str(prop_id))


def getter_by_column(column_name):
    # type: (str) -> Callable[[Table], Set[str]]
    def func(table):
        # type: (Table) -> Set[str]
        return set(table[column_name])
    return func

get_obsids = getter_by_column(
    'obsid')  # type: Callable[[Table], Set[str]]

get_proposal_ids = getter_by_column(
    'proposal_id')  # type: Callable[[Table], Set[str]]
