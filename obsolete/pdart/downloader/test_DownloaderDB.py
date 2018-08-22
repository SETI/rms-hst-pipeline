import unittest

from pdart.downloader.DownloaderDB \
    import _str_set_to_string, _string_to_str_set


class Test_DownloaderDB(unittest.TestCase):
    def test_str_set_to_string(self):
        # type: () -> None
        self.assertEqual('', _str_set_to_string(set()))
        self.assertEqual(
            'hst_00001 hst_00002 hst_00004 hst_00006 hst_00009',
            _str_set_to_string(set(
                    ['hst_00009', 'hst_00004', 'hst_00006',
                     'hst_00002', 'hst_00001'])))

    def test_string_to_str_set(self):
        # type: () -> None
        self.assertEqual(set(), _string_to_str_set(''))
        self.assertEqual(
            set(['hst_00009', 'hst_00004', 'hst_00006',
                 'hst_00002', 'hst_00001']),
            _string_to_str_set(
                'hst_00001 hst_00002 hst_00006 hst_00004 hst_00009'))
