import abc
import xml.dom


class HExc(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def to_xml(self, document):
        pass

    def print_as_xml(self):
        document = xml.dom.getDOMImplementation().createDocument(None,
                                                                 None,
                                                                 None)
        xml_ = self.to_xml(document)
        document.appendChild(xml_)
        print xml_.toprettyxml(indent='  ', newl='\n', encoding='utf-8')


class HException(HExc):
    def __init__(self, exception, stack_trace):
        assert exception
        self.exception = exception

        assert stack_trace
        self.stack_trace = stack_trace

    def __repr__(self):
        return 'HException(%r, %r)' % (self.exception, self.stack_trace)

    def __str__(self):
        return 'HException(%s, %s)' % (self.exception, self.stack_trace)

    def to_xml(self, document):
        res = document.createElement('HException')
        res.appendChild(document.createTextNode(self.stack_trace))
        return res


class HExceptionGroup(HExc):
    def __init__(self, label, exceptions):
        # label may be None.
        self.label = label
        # exceptions is a list of HExcs.  (An un-Pythonic type
        # check follows.)
        assert exceptions
        assert isinstance(exceptions, list)
        for ex in exceptions:
            assert isinstance(ex, HExc)
        self.exceptions = exceptions

    def __repr__(self):
        return 'HExceptionGroup(%r, %r)' % (self.label, self.exceptions)

    def __str__(self):
        return 'HExceptionGroup(%s, %s)' % (self.label, self.exceptions)

    def to_xml(self, document):
        res = document.createElement('HExceptionGroup')
        if self.label:
            res.setAttribute('label', self.label)
        for ex in self.exceptions:
            child = ex.to_xml(document)
            res.appendChild(child)
        return res

# import traceback
# import unittest

# class TestHException(unittest.TestCase):
#     def test_x(self):
#         try:
#             raise Exception('foo')
#         except Exception as e:
#             ex = HException(e, traceback.format_exc())
#         eg = HExceptionGroup('bar', [ex])
#         ex.print_as_xml()
#         eg.print_as_xml()
#
#         print str(ex)
#         print str(eg)
#
#         print repr(ex)
#         print repr(eg)
#
# if __name__ == '__main__':
#     unittest.main()
