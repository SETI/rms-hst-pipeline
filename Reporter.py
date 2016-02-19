import abc
import csv


class Reporter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def report(self, pass_, context, msg, tag):
        pass

    def beginReporting(self):
        pass

    def endReporting(self):
        pass


class StdoutReporter(Reporter):
    def report(self, pass_, context, msg, tag):
        print pass_ + ':',
        print context + ':',
        if tag is not None:
            print msg, tag
        else:
            print msg


class CsvReporter(Reporter):
    def __init__(self, filepath):
        assert filepath
        self.filepath = filepath
        self.file = None

    def report(self, pass_, context, msg, tag):
        if tag is None:
            tag = ''
        self.csvWriter.writerow([pass_, context, msg, tag])

    def beginReporting(self):
        self.file = open(self.filepath, 'wb')
        self.csvWriter = csv.writer(self.file)
        # Write the headers.
        self.csvWriter.writerow(['PASS', 'CONTEXT', 'MESSAGE', 'TAG'])

    def endReporting(self):
        self.file.close()
        self.csvWriter = None
        self.file = None
