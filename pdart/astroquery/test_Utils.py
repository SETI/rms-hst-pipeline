from typing import TYPE_CHECKING
import unittest

from pdart.astroquery.Utils import filter_table, ymdhms_format_from_mjd

from astropy.table.row import Row


class TestUtils(unittest.TestCase):
    def test_sanity(self) -> None:
        """
        Just a simple sanity check on converting between MJD and
        dates.
        """
        ymdhms = ymdhms_format_from_mjd(57986.72013889)
        self.assertEqual("2017-08-21T17:17:00", ymdhms)

    def test_filter_table(self) -> None:

        # I'm using MastSlice to get a table, but it would probably be
        # better to hand-construct one.
        from pdart.astroquery.Astroquery import MastSlice

        start_date = (2018, 1, 1)
        end_date = (2018, 3, 26)
        slice = MastSlice(start_date, end_date)
        proposal_id = int(slice.observations_table[0]["proposal_id"])

        def proposal_id_matches(row: Row) -> bool:
            return int(row["proposal_id"]) == proposal_id

        proposal_table = filter_table(proposal_id_matches, slice.observations_table)

        # We know that it drops at least the first row because that's
        # where we got the proposal_id from.
        self.assertTrue(len(proposal_table) < len(slice.observations_table))

        # Make sure we got what we filtered for.
        for row in proposal_table:
            self.assertEqual(proposal_id, int(row["proposal_id"]))
