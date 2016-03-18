import os
import subprocess
import tempfile
import unittest
import xml.dom.minidom


XML_SCHEMA = './xml/PDS4_PDS_1500.xsd.xml'
SCHEMATRON_SCHEMA = './xml/PDS4_PDS_1500.sch.xml'


def _run_subprocess(cmd, stdin=None):
    """
    Run a command in a subprocess.  Command can be either a string
    with the command name, or an array of the command name and its
    arguments.  If stdin is given, use it as input.  Return a triple
    of exit_code, stderr, and stdout.  (NOTE: alphabetical order.)
    The last two are strings.
    """
    if stdin is None:
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
    else:
        p = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate(stdin)
    exit_code = p.returncode
    # return in alphabetical order
    return (exit_code, stderr, stdout)


def _xmllint(filepath, schema=XML_SCHEMA, stdin=None):
    """
    Run xmllint on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns a
    triple of exit_code, stderr and stdout.
    """
    if stdin is None:
        return _run_subprocess(['xmllint', '--noout',
                                '--schema', schema, filepath])
    else:
        return _run_subprocess(['xmllint', '--noout', '--schema', schema, '-'],
                               stdin=stdin)


def xml_schema_failures(filepath, schema=XML_SCHEMA, stdin=None):
    """
    Run xmllint on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns None if
    there are no failures; returns a string containing the failures if
    they exist.
    """
    exit_code, stderr, _stdout = _xmllint(filepath, schema=schema, stdin=stdin)
    if exit_code == 0:
        return None
    else:
        # ignore stdout
        assert stderr
        # TODO The replace probably ought to be done only in the
        # reporting, only as necessary
        return stderr.replace('\n', '\\n')


def _probatron(filepath, schema=SCHEMATRON_SCHEMA):
    """
    Run probatron on the XML at the filepath validating against the
    schema.  Returns a triple of exit_code, stderr and stdout.
    """
    return _run_subprocess(['java', '-jar', 'probatron.jar',
                            '-r0',  # output report as terse SVRL
                            '-n1',  # emit line/col numbers in report
                            filepath, schema])


def _probatron_with_stdin(filepath, schema=SCHEMATRON_SCHEMA, stdin=None):
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns a
    triple of exit_code, stderr and stdout.
    """
    if stdin is None:
        return _probatron(filepath, schema)
    else:
        (handle, filepath) = tempfile.mkstemp()
        try:
            f = os.fdopen(handle, 'w')
            f.write(stdin)
            f.close()
            return _probatron(filepath, schema)
        finally:
            os.remove(filepath)


def _probatron_with_svrl_result(filepath,
                                schema=SCHEMATRON_SCHEMA,
                                stdin=None):
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns the
    SVRL as XML.
    """
    exit_code, stderr, stdout = _probatron_with_stdin(filepath, schema, stdin)
    assert exit_code == 0, 'exit_code = %r' % exit_code
    assert stderr == '', 'stderr = %r' % stderr
    return xml.dom.minidom.parseString(stdout)


def _svrl_failures(svrl):
    return svrl.documentElement.getElementsByTagName('svrl:failed-assert')


def _svrl_has_failures(svrl):
    return len(_svrl_failures(svrl)) > 0


def schematron_failures(filepath, schema=SCHEMATRON_SCHEMA, stdin=None):
    """
    Run probatron on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns None if
    there are no failures; returns a string containing the failures if
    they exist.
    """
    svrl = _probatron_with_svrl_result(filepath, schema, stdin)
    failures = _svrl_failures(svrl)
    if len(failures) > 0:
        # should I have a pretty option here for human-readability?

        # TODO The replace probably ought to be done only in the
        # reporting, only as necessary
        return ('\n'.join([f.toxml() for f in failures])).replace('\n', '\\n')
    else:
        return None


class TestXmlSchema(unittest.TestCase):
    # Exploratory testing: I want to use external programs that aren't
    # well documented as part of my code, so I'm using unittests to
    # explore the behavior of the system.
    def test__run_subprocess(self):
        exit_code, stderr, stdout = _run_subprocess('ls')
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        exit_code, stderr, stdout = _run_subprocess(['echo', 'foo', 'bar'])
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertEquals('foo bar\n', stdout)

        stdin = 'foo bar\nbaz\n'
        exit_code, stderr, stdout = _run_subprocess('cat', stdin)
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertEquals(stdin, stdout)

        exit_code, stderr, stdout = \
            _run_subprocess(['cat', 'this_file_does_not_exist'])
        self.assertNotEquals(0, exit_code)
        self.assertNotEquals('', stderr)
        self.assertEquals('', stdout)

    def test_run_xml_schema(self):
        # Correct XML, but doesn't match the schema
        failures = \
            xml_schema_failures('-',
                                stdin='<library><book/><book/></library>')
        self.assertIsNotNone(failures)
        self.assertNotIn('\n', failures)

        # Valid XML according to the schema.
        failures = xml_schema_failures('./testfiles/bundle.xml')
        self.assertIsNone(failures)

    def test_run_schematron(self):
        exit_code, stderr, stdout = _probatron('./testfiles/bundle.xml')
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        exit_code, stderr, stdout = \
            _probatron_with_stdin(None, SCHEMATRON_SCHEMA,
                                  stdin='<library><book/><book/></library>')
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        svrl = _probatron_with_svrl_result('./testfiles/bundle.xml')
        self.assertFalse(_svrl_has_failures(svrl))

        svrl = _probatron_with_svrl_result('./testfiles/bad_bundle.xml')
        self.assertTrue(_svrl_has_failures(svrl))

        self.assertIsNone(schematron_failures('./testfiles/bundle.xml'))
        failures = schematron_failures('./testfiles/bad_bundle.xml')
        self.assertIsNotNone(failures)
        self.assertNotIn('\n', failures)


if __name__ == '__main__':
    unittest.main()
