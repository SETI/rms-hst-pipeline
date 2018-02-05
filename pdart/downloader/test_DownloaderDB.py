import unittest

from pdart.downloader.DownloaderDB \
    import _int_set_to_string, _string_to_int_set


class Test_DownloaderDB(unittest.TestCase):
    def test_int_set_to_string(self):
        # type: () -> None
        self.assertEqual('', _int_set_to_string(set()))
        self.assertEqual('1 2 4 6 9',
                         _int_set_to_string(set([9, 4, 6, 2, 1])))

    def test_string_to_int_set(self):
        # type: () -> None
        self.assertEqual(set(), _string_to_int_set(''))
        self.assertEqual(set([9, 4, 6, 2, 1]), _string_to_int_set('1 2 6 4 9'))
