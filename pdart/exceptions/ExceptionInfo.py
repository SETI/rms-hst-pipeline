"""
For Python to handle multiple implementations of functions and the
multiple exceptions they might raise, we need to capture the
information from those exceptions.  Normally, Python does not retain
this information and just displays the current state when reporting an
exception, then immediately forgets it.

The :py:mod:`pdart.exception.ExceptionInfo` module defines an
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


class CalculationException(Exception):
    """
    An :exc:`Exception` carrying
    :class:`~pdart.exception.ExceptionInfo.ExceptionInfo`
    """

    def __init__(self, msg, exception_info):
        assert isinstance(exception_info, ExceptionInfo)
        Exception.__init__(self, msg)
        self.exception_info = exception_info

    def __str__(self):
        return 'CalculationException(%s)' % self.message

    def __repr__(self):
        return 'CalculationException(%r, %r)' % (self.message,
                                                 self.exception_info)


class ExceptionInfo(object):
    """
    An abstract class wrapping either a single exception and stack
    trace or a labeled group of
    :class:`~pdart.exception.ExceptionInfo.ExceptionInfo`.
    """
    __metaclass__ = abc.ABCMeta

    def to_pretty_xml(self):
        """
        Return human-readable XML text for this
        :class:`~pdart.exception.ExceptionInfo.ExceptionInfo.`
        """
        return self.to_xml().toprettyxml()

    def to_xml(self):
        """
        Return an XML document data structure for this
        :class:`~pdart.exception.ExceptionInfo.ExceptionInfo`.
        """
        document = xml.dom.getDOMImplementation().createDocument(None,
                                                                 None,
                                                                 None)
        elmt = self.to_xml_fragment(document)
        document.appendChild(elmt)
        return document

    @abc.abstractmethod
    def to_xml_fragment(self, document):
        """
        Return an XML data structure for this
        :class:`~pdart.exception.ExceptionInfo.ExceptionInfo`.
        """
        pass


class SingleExceptionInfo(ExceptionInfo):
    """
    A class wrapping a single :exc:`Exception` and its stack trace.
    """
    def __init__(self, exception, stack_trace):
        self.exception = exception
        self.stack_trace = stack_trace

    def to_xml_fragment(self, document):
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
    :class:`~pdart.exception.ExceptionInfo.ExceptionInfo`.
    """
    def __init__(self, label, exception_infos):
        assert isinstance(label, str)
        self.label = label
        assert isinstance(exception_infos, list)
        self.exception_infos = exception_infos

    def to_xml_fragment(self, document):
        res = document.createElement('GroupedExceptionInfo')
        label = document.createElement('label')
        label_text = document.createTextNode(self.label)
        label.appendChild(label_text)
        for exception_info in self.exception_infos:
            res.appendChild(exception_info.to_xml_fragment(document))
        return res

    def __repr__(self):
        return 'GroupedExceptionInfo(%r, %r)' % \
            (self.label, self.exception_infos)

    def __str__(self):
        return 'GroupedExceptionInfo(%r, %s)' % \
            (self.label, self.exception_infos)
