import abc
import xml.dom


class CalculationException(Exception):
    """An :class:`Exception` carrying :class:`ExceptionInfo`"""

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
    An abstract class wrapping either a single :class:`Exception` or a
    labeled group of :class:`Exception`.
    """
    __metaclass__ = abc.ABCMeta

    def to_pretty_xml(self):
        """
        Return human-readable XML text for this
        :class:`ExceptionInfo.`
        """
        return self.to_xml().toprettyxml()

    def to_xml(self):
        """
        Return an XML document data structure for this
        :class:`ExceptionInfo`.
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
        Return an XML data structure for this :class:`ExceptionInfo`.
        """
        pass


class SingleExceptionInfo(ExceptionInfo):
    """
    A class wrapping a single :class:`Exception`.
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
    A class wrapping a labeled group of :class:`Exception`.
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
