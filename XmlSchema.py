import subprocess
import unittest


def run_subprocess(cmd, stdin=None):
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


XML_SCHEMA = './xml/PDS4_PDS_1500.xsd.xml'


def xmllint(filepath, schema=XML_SCHEMA, stdin=None):
    """
    Run xmllint on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns a
    triple of exit_code, stderr and stdout.
    """
    if stdin is None:
        return run_subprocess(['xmllint', '--noout',
                               '--schema', schema, filepath])
    else:
        return run_subprocess(['xmllint', '--noout', '--schema', schema, '-'],
                              stdin=stdin)


def xml_schema_failures(filepath, schema=XML_SCHEMA, stdin=None):
    """
    Run xmllint on the XML at the filepath (ignored if stdin is not
    None) or in stdin, validating against the schema.  Returns None if
    there are no failures; returns a string containing the failures if
    they exist.
    """
    exit_code, stderr, stdout = xmllint(filepath, schema=schema, stdin=stdin)
    if exit_code == 0:
        return None
    else:
        # ignore stdout
        assert stderr
        return stderr


class TestXmlSchema(unittest.TestCase):
    # Exploratory testing: I want to use an extern program as part of
    # my code, so I'm using unittests to explore the behavior of the
    # system.
    def test_run_subprocess(self):
        exit_code, stderr, stdout = run_subprocess('ls')
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertNotEquals('', stdout)

        exit_code, stderr, stdout = run_subprocess(['echo', 'foo', 'bar'])
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertEquals('foo bar\n', stdout)

        stdin = 'foo bar\nbaz\n'
        exit_code, stderr, stdout = run_subprocess('cat', stdin)
        self.assertEquals(0, exit_code)
        self.assertEquals('', stderr)
        self.assertEquals(stdin, stdout)

        exit_code, stderr, stdout = \
            run_subprocess(['cat', 'this_file_does_not_exist'])
        self.assertNotEquals(0, exit_code)
        self.assertNotEquals('', stderr)
        self.assertEquals('', stdout)

        # Correct XML, but doesn't match the schema
        res = xml_schema_failures('-',
                                  stdin='<library><book/><book/></library>')
        self.assertIsNotNone(res)

        # Valid XML according to the schema.
        VALID_XML_FILEPATH = './testfiles/bundle.xml'
        res = xml_schema_failures(VALID_XML_FILEPATH)
        self.assertIsNone(res)

if __name__ == '__main__':
    unittest.main()
