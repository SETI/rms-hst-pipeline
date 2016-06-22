import sys

from pdart.xml.Schema import *

if __name__ == '__main__':
    if len(sys.argv) is not 2:
        sys.stderr.write('usage: python ValidateLabel.py <pds4 label>\n')
        sys.exit(1)

    label_filepath = sys.argv[1]
    # print ('label_filepath is %s' % label_filepath)

    failures = xml_schema_failures(label_filepath)
    # print ('xml_schema_failures(%s) = %s' % (label_filepath, failures))
    if failures is not None:
        print 'xml_schema_failures():'
        print failures
        sys.exit(1)

    failures = schematron_failures(label_filepath)
    # print ('schematron_failures(%s) = %s' % (label_filepath, failures))
    if failures is not None:
        print 'schematron_failures():'
        print failures
        sys.exit(1)
    else:
        sys.exit(0)
