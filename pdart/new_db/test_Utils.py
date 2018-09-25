# coding=utf-8
import os
import tempfile
from typing import TYPE_CHECKING
import unittest

from pdart.new_db.Utils import file_md5, string_md5, unicode_md5


class Test_Utils(unittest.TestCase):
    def test_file_md5(self):
        (handle, filepath) = tempfile.mkstemp()
        try:
            os.close(handle)
            self.assertEqual('d41d8cd98f00b204e9800998ecf8427e',
                             file_md5(filepath))
        finally:
            os.remove(filepath)

    def test_string_md5(self):
        self.assertEqual('d41d8cd98f00b204e9800998ecf8427e',
                         string_md5(''))
        self.assertEqual('3858f62230ac3c915f300c664312c63f',
                         string_md5('foobar'))

    def test_unicode_md5(self):
        self.assertEqual('d41d8cd98f00b204e9800998ecf8427e',
                         unicode_md5(u''))
        self.assertEqual('11f8641f0a76f703b6e905dd7ed9713b',
                         unicode_md5(u'¿Cómo está Ud., señor?'))
        self.assertEqual('8277e4c295daf69388600e4d3befe35f',
                         unicode_md5(u'पाइथन मन पर््दैना ।'))
