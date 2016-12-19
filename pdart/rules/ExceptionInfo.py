"""
For Python to handle multiple implementations of functions and the
multiple exceptions they might raise, we need to capture the
information from those exceptions.  Normally, Python does not retain
this information and just displays the current state when reporting an
exception, then immediately forgets it.

The :py:mod:`pdart.rules.ExceptionInfo` module defines an
exception, :class:`CalculationException`, that can hold this
information.

**New to PDART?** You don't need to understand the internals of this
package to use it effectively.  Explanation of internals follows.

For any exception raised, we need to keep the exception itself and the
stack trace active at the time of the raising.  We call this
:class:`ExceptionInfo`.

If a normal Python function exception is raised, we capture the
information in a :class:`SingleExceptionInfo`.  If we've composed
multiple implementations into a single function (also called a
*rule*), multiple exceptions may have been raised.  In that case, we
capture the information in a :class:`GroupedExceptionInfo` which
contains both the exception info and a label to show the function
whose implementations raised them.

In this way, exception information forms a tree structure, where each
node is either a single exception (those are the leaf nodes) or a
labeled set of exception info (those are the branch nodes).  Each
exception in a labeled set may then itself be either single or
multiple, permitting arbitrarily deep nesting.

Since our :class:`ExceptionInfo` is itself not a Python exception, we
need our own exception class to hold the info, the
:class:`CalculationException`.

Using these classes, we can raise multiple exceptions for arbitrarily
complex calculations and retain the full information needed to debug
the failures.
"""
import abc
import xml.dom

from typing import cast, List, TYPE_CHECKING
if TYPE_CHECKING:
    import xml.dom.minidom


class ExceptionInfo(object):
    """
    An abstract class wrapping either a single exception and stack
    trace or a labeled group of
    :class:`~pdart.rules.ExceptionInfo.ExceptionInfo`.
    """
    __metaclass__ = abc.ABCMeta

    def to_pretty_xml(self):
        # type: () -> unicode
        """
        Return human-readable XML text for this
        :class:`~pdart.rules.ExceptionInfo.ExceptionInfo.`
        """
        return self.to_xml().toprettyxml()

    def to_xml(self):
        # type: () -> xml.dom.minidom.Element
        """
        Return an XML document data structure for this
        :class:`~pdart.rules.ExceptionInfo.ExceptionInfo`.
        """
        # type: () -> unicode
        document = xml.dom.getDOMImplementation().createDocument(None,
                                                                 None,
                                                                 None)
        elmt = self.to_xml_fragment(document)
        document.appendChild(elmt)
        return document

    @abc.abstractmethod
    def to_xml_fragment(self, document):
        # type: (xml.dom.minidom.Document) -> xml.dom.minidom.Element
        """
        Return an XML data structure for this
        :class:`~pdart.rules.ExceptionInfo.ExceptionInfo`.
        """
        pass


_EXC_INFOS = List[ExceptionInfo]


class CalculationException(Exception):
    """
    An :exc:`Exception` carrying
    :class:`~pdart.rules.ExceptionInfo.ExceptionInfo`
    """

    def __init__(self, msg, exception_info):
        # type: (unicode, ExceptionInfo) -> None
        assert isinstance(exception_info, ExceptionInfo)
        Exception.__init__(self, msg)
        self.exception_info = exception_info

    def __str__(self):
        return 'CalculationException(%s)' % self.message

    def __repr__(self):
        return 'CalculationException(%r, %r)' % (self.message,
                                                 self.exception_info)


class SingleExceptionInfo(ExceptionInfo):
    """
    A class wrapping a single :exc:`Exception` and its stack trace.
    """
    def __init__(self, exception, stack_trace):
        # type: (Exception, str) -> None
        self.exception = exception
        self.stack_trace = stack_trace

    def to_xml_fragment(self, document):
        # type: (xml.dom.minidom.Document) -> xml.dom.minidom.Element
        res = document.createElement('SingleExceptionInfo')
        message = document.createElement('message')
        message_text = document.createTextNode(self.exception.message)
        message.appendChild(message_text)
        res.appendChild(message)

        stack_trace = document.createElement('stack_trace')
        stack_trace_text = document.createTextNode(self.stack_trace)
        stack_trace.appendChild(stack_trace_text)
        res.appendChild(stack_trace)

        return res

    def __repr__(self):
        return 'SimpleExceptionInfo(%r, %r)' % (self.exception,
                                                self.stack_trace)

    def __str__(self):
        return 'SimpleExceptionInfo(%s)' % str(self.exception)


class GroupedExceptionInfo(ExceptionInfo):
    """
    A class wrapping a labeled group of
    :class:`~pdart.rules.ExceptionInfo.ExceptionInfo`.
    """
    def __init__(self, label, exception_infos):
        # type: (unicode, _EXC_INFOS) -> None

        self.label = label
        self.exception_infos = exception_infos

    def to_xml_fragment(self, document):
        # type: (xml.dom.minidom.Document) -> xml.dom.minidom.Element
        res = document.createElement('GroupedExceptionInfo')

        label_elmt = document.createElement('label')

        label_text = document.createTextNode(self.label)

        label_elmt.appendChild(label_text)
        for exception_info in self.exception_infos:
            res.appendChild(exception_info.to_xml_fragment(document))
        return res

    def __repr__(self):
        return 'GroupedExceptionInfo(%r, %r)' % \
            (self.label, self.exception_infos)

    def __str__(self):
        return 'GroupedExceptionInfo(%r, %s)' % \
            (self.label, self.exception_infos)
