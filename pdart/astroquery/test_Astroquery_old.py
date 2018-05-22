import unittest
from pdart.astroquery.Astroquery_old import *
from pdart.astroquery.Utils import \
    table_to_list_of_dicts, ymdhms_format_from_mjd


class TestAstroquery_old(unittest.TestCase):
    @unittest.skip('work in progress')
    def test_get_hst_moving_images(self):
        # type: () -> None
        xxxs = get_hst_moving_images(-10000)
        self.assertTrue(len(xxxs) > 0)
        print type(xxxs)
        d = table_to_list_of_dicts(xxxs)[0]
        for k, v in sorted(d.items()):
            print k, v
        # I'm interested in: dataURL, obsid
        # proposal_id, t_obs_release
        print len(get_obsids(xxxs))
        prop_ids = get_proposal_ids(xxxs)
        print sorted(int(id) for id in prop_ids)
        print len(prop_ids)
        print ymdhms_format_from_mjd(0)
        print ymdhms_format_from_mjd(-10000)

        assert False
