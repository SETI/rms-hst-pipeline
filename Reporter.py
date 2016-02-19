import abc


class Reporter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def report(self, pass_, context, msg, tag):
        pass


class StdoutReporter(Reporter):
    def report(self, pass_, context, msg, tag):
        print pass_ + ':',
        print context + ':',
        if tag is not None:
            print msg, tag
        else:
            print msg


class OtherReporter(Reporter):
    def report(self, pass_, context, msg, tag):
        print pass_ + '#',
        print context + '#',
        if tag is not None:
            print msg, '#', tag
        else:
            print msg
