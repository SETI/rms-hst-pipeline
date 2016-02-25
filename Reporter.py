import abc
import csv


class Reporter(object):
    """An abstract class for Passes to report their results through."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def report(self, pass_, context, msg, tag):
        """Report the results."""
        pass

    def begin_reporting(self):
        """
        Prepare to report.  May open a file or initialize a UI,
        depending on the Reporter implementation.
        """
        pass

    def end_reporting(self):
        """
        Complete reporting.  May close a file or do any other work,
        depending on the Reporter implementation.
        """
        pass


class StdoutReporter(Reporter):
    """A reporter that reports by writing to stdout."""

    def report(self, pass_, context, msg, tag):
        """Report by writing a line to stdout."""
        print pass_ + ':',
        print context + ':',
        if tag is not None:
            print msg, tag
        else:
            print msg


class CsvReporter(Reporter):
    """
    A reporter that reports by writing to a file of comma-separated
    values that can be opened into a spreadsheet.
    """

    def __init__(self, filepath):
        """Create the reporter using the given filepath for the CSV file."""
        assert filepath
        self.filepath = filepath
        self.file = None

    def report(self, pass_, context, msg, tag):
        """Write a row into the CSV file."""
        if tag is None:
            tag = ''
        self.csv_writer.writerow([pass_, context, msg, tag])

    def begin_reporting(self):
        """Open the CSV file and write the header row."""
        self.file = open(self.filepath, 'wb')
        self.csv_writer = csv.writer(self.file)
        # Write the headers.
        self.csv_writer.writerow(['PASS', 'CONTEXT', 'MESSAGE', 'TAG'])

    def end_reporting(self):
        """Close the CSV file."""
        self.file.close()
        self.csv_writer = None
        self.file = None
