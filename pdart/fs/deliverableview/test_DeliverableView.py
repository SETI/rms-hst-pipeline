import unittest

from pdart.fs.deliverableview.DeliverableView import translate_filepath


class TestDeliverableView(unittest.TestCase):
    def test_translate_filepath(self) -> None:
        # top-level
        self.assertEqual("/", translate_filepath("/"))

        # bundle-level
        self.assertEqual("/hst_12345", translate_filepath("/hst_12345$"))
        self.assertEqual(
            "/hst_12345/foo/bar/file.txt",
            translate_filepath("/hst_12345$/foo/bar/file.txt"),
        )

        # collection-level
        self.assertEqual(
            "/hst_12345/data_acs_raw", translate_filepath("/hst_12345$/data_acs_raw$")
        )
        self.assertEqual(
            "/hst_12345/data_acs_raw/foo/bar/file.txt",
            translate_filepath("/hst_12345$/data_acs_raw$/foo/bar/file.txt"),
        )

        # product-level

        ## document
        self.assertEqual(
            "/hst_12345/document/phase2",
            translate_filepath("/hst_12345$/document$/phase2$"),
        )

        self.assertEqual(
            "/hst_12345/document/phase2/phase2.pdf",
            translate_filepath("/hst_12345$/document$/phase2$/phase2.pdf"),
        )

        ## data
        self.assertEqual(
            "/hst_12345/data_acs_raw/visit_01/j6gp01lzq_raw.xml",
            translate_filepath(
                "/hst_12345$/data_acs_raw$/j6gp01lzq_raw$/j6gp01lzq_raw.xml"
            ),
        )

        with self.assertRaises(ValueError):
            translate_filepath(
                "/hst_12345$/data_acs_raw$/j6gp01lzq_raw$/"
                "john$/jacob$/jingleheimer$/schmidt$"
            )
