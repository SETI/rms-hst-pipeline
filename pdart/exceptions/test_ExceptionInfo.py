import unittest
import xml.dom

from pdart.exceptions.ExceptionInfo import *


def test_ExceptionInfo():
    try:
        ce = CalculationException('<msg>', Exception('foo'))
        self.assertFalse(True)
    except AssertionError:
        pass


class TestSingleExceptionInfo(unittest.TestCase):
    def test_to_xml(self):
        se = SingleExceptionInfo(Exception('test'), '{stack trace}')
        xml_ = se.to_xml()
        self.assertTrue(isinstance(xml_, xml.dom.minidom.Document))

    def test_to_pretty_xml(self):
        se = SingleExceptionInfo(Exception('test'), '{stack trace}')
        xml_ = se.to_pretty_xml()
        xml_ = xml_.split('\n')
        self.assertEquals('<SingleExceptionInfo>', xml_[1])


class TestGroupedExceptionInfo(unittest.TestCase):
    def test_to_xml(self):
        ses = [SingleExceptionInfo(Exception('test1'), '{stack trace}'),
               SingleExceptionInfo(Exception('test2'), '{stack trace}'),
               SingleExceptionInfo(Exception('test3'), '{stack trace}')]
        ge = GroupedExceptionInfo('foo', ses)
        xml_ = ge.to_xml()
        self.assertTrue(isinstance(xml_, xml.dom.minidom.Document))

    def test_to_pretty_xml(self):
        ses = [SingleExceptionInfo(Exception('test1'), '{stack trace}'),
               SingleExceptionInfo(Exception('test2'), '{stack trace}'),
               SingleExceptionInfo(Exception('test3'), '{stack trace}')]
        ge = GroupedExceptionInfo('bar', ses)
        xml_ = ge.to_pretty_xml()
        xml_ = xml_.split('\n')
        self.assertEquals('<GroupedExceptionInfo>', xml_[1])
        self.assertEquals('<SingleExceptionInfo>', xml_[2][1:])
