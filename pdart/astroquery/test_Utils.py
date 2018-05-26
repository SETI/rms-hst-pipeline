import unittest

from pdart.astroquery.Utils import *

if TYPE_CHECKING:
    from astroquery.table.row import Row


class TestUtils(unittest.TestCase):
    def test_sanity(self):
        # type: () -> None
        """
        Just a simple sanity check on converting between MJD and
        dates.
        """
        ymdhms = ymdhms_format_from_mjd(57986.72013889)
        self.assertEqual('2017-08-21T17:17:00', ymdhms)

    def test_filter_table(self):
        from pdart.astroquery.Astroquery import MastSlice
        start_date = (1900, 1, 1)
        end_date = (2018, 3, 26)
        slice = MastSlice(start_date, end_date)
        proposal_id = slice.observations_table[0]['proposal_id']

        def proposal_id_matches(row):
            # test: (Row) -> bool
            return int(row['proposal_id']) == proposal_id

        proposal_table = filter_table(proposal_id_matches,
                                      slice.observations_table)

        self.assertTrue(len(proposal_table) < len(slice.observations_table))
