# coding=utf-8
import os
import tempfile
import unittest

from pdart.db.utils import file_md5, string_md5


class Test_utils(unittest.TestCase):
    def test_file_md5(self) -> None:
        (handle, filepath) = tempfile.mkstemp()
        try:
            os.close(handle)
            self.assertEqual("d41d8cd98f00b204e9800998ecf8427e", file_md5(filepath))
        finally:
            os.remove(filepath)

    def test_string_md5(self) -> None:
        self.assertEqual("d41d8cd98f00b204e9800998ecf8427e", string_md5(""))
        self.assertEqual("3858f62230ac3c915f300c664312c63f", string_md5("foobar"))
        self.assertEqual("d41d8cd98f00b204e9800998ecf8427e", string_md5(""))
        self.assertEqual(
            "11f8641f0a76f703b6e905dd7ed9713b", string_md5("¿Cómo está Ud., señor?")
        )
        self.assertEqual(
            "8277e4c295daf69388600e4d3befe35f", string_md5("पाइथन मन पर््दैना ।")
        )
