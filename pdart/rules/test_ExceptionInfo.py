import unittest
import xml.dom

from pdart.rules.ExceptionInfo import ExceptionInfo, GroupedExceptionInfo, \
    SingleExceptionInfo


class TestSingleExceptionInfo(unittest.TestCase):
    def test_to_xml(self):
        # type: () -> None
        se = SingleExceptionInfo(Exception('test'), '{stack trace}')
        xml_ = se.to_xml()
        self.assertTrue(isinstance(xml_, xml.dom.minidom.Document))

    def test_to_pretty_xml(self):
        # type: () -> None
        se = SingleExceptionInfo(Exception('test'), '{stack trace}')
        xml_ = se.to_pretty_xml().split('\n')
        self.assertEquals('<SingleExceptionInfo>', xml_[1])


class TestGroupedExceptionInfo(unittest.TestCase):
    def test_to_xml(self):
        # type: () -> None
        ses = [SingleExceptionInfo(Exception('test1'), '{stack trace}'),
               SingleExceptionInfo(Exception('test2'), '{stack trace}'),
               SingleExceptionInfo(Exception('test3'), '{stack trace}')]
        # type: List[ExceptionInfo]
        ge = GroupedExceptionInfo('foo', ses)
        xml_ = ge.to_xml()
        self.assertTrue(isinstance(xml_, xml.dom.minidom.Document))

    def test_to_pretty_xml(self):
        # type: () -> None
        ses = [SingleExceptionInfo(Exception('test1'), '{stack trace}'),
               SingleExceptionInfo(Exception('test2'), '{stack trace}'),
               SingleExceptionInfo(Exception('test3'), '{stack trace}')]
        # type: List[ExceptionInfo]
        ge = GroupedExceptionInfo('bar', ses)
        xml_ = ge.to_pretty_xml().split('\n')
        self.assertEquals('<GroupedExceptionInfo>', xml_[1])
        self.assertEquals('<SingleExceptionInfo>', xml_[2][1:])
