from astroquery.mast import Observations
from astropy.table import Table
from pdart.astroquery.astroquery import _YMD

start_date: _YMD = (2000, 1, 1)
end_date: _YMD = (2001, 1, 1)
proposal_id: int = 9296


def experiment() -> None:
    table: Table = Observations.query_criteria(
        dataproduct_type=["image"],
        dataRights="PUBLIC",
        obs_collection=["HST"],
        proposal_id=proposal_id,
        t_obs_release=(start_date, end_date),
        mtFlag=True,
    )
    print(f"{table}")


if __name__ == "__main__":
    experiment()
