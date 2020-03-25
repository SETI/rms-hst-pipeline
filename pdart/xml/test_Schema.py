import unittest

from pdart.xml.Schema import (
    PDS_SCHEMATRON_SCHEMA,
    probatron,
    probatron_with_stdin,
    probatron_with_svrl_result,
    run_subprocess,
    schematron_failures,
    svrl_failures,
    svrl_has_failures,
    xml_schema_failures,
)
from pdart.xml.Utils import path_to_testfile


class TestXmlSchema(unittest.TestCase):
    # Exploratory testing: I want to use external programs that aren't
    # well documented as part of my code, so I'm using unittests to
    # explore the behavior of the system.
    def test_run_subprocess(self) -> None:
        exit_code, stderr, stdout = run_subprocess("ls")
        self.assertEqual(0, exit_code)
        self.assertEqual(b"", stderr)
        self.assertNotEqual(b"", stdout)

        exit_code, stderr, stdout = run_subprocess(["echo", "foo", "bar"])
        self.assertEqual(0, exit_code)
        self.assertEqual(b"", stderr)
        self.assertEqual(b"foo bar\n", stdout)

        stdin = b"foo bar\nbaz\n"
        exit_code, stderr, stdout = run_subprocess("cat", stdin)
        self.assertEqual(0, exit_code)
        self.assertEqual(b"", stderr)
        self.assertEqual(stdin, stdout)

        exit_code, stderr, stdout = run_subprocess(["cat", "this_file_does_not_exist"])
        self.assertNotEqual(0, exit_code)
        self.assertNotEqual(b"", stderr)
        self.assertEqual(b"", stdout)

    def test_run_xml_schema(self) -> None:
        # Correct XML, but doesn't match the schema
        failures = xml_schema_failures("-", stdin=b"<library><book/><book/></library>")
        self.assertIsNotNone(failures)

        # Valid XML according to the schema.
        failures = xml_schema_failures(path_to_testfile("bundle.xml"))
        self.assertIsNone(failures)

    def test_run_schematron(self) -> None:
        exit_code, stderr, stdout = probatron(path_to_testfile("bundle.xml"))
        self.assertEqual(0, exit_code)
        self.assertEqual(b"", stderr)
        self.assertNotEqual(b"", stdout)

        exit_code, stderr, stdout = probatron_with_stdin(
            "<test_Schema.test_run_schematron>",
            stdin=b"<library><book/><book/></library>",
            schema=PDS_SCHEMATRON_SCHEMA,
        )
        self.assertEqual(0, exit_code)
        self.assertEqual(b"", stderr)
        self.assertNotEqual(b"", stdout)

        svrl = probatron_with_svrl_result(path_to_testfile("bundle.xml"))
        failures = svrl_failures(svrl)
        self.assertFalse(
            failures, ("\n".join([f.toxml() for f in failures])).replace("\n", "\\n")
        )

        svrl = probatron_with_svrl_result(path_to_testfile("bad_bundle.xml"))
        failures = svrl_failures(svrl)
        self.assertTrue(failures)

        self.assertIsNone(schematron_failures(path_to_testfile("bundle.xml")))
        sch_failures = schematron_failures(path_to_testfile("bad_bundle.xml"))
        self.assertNotEqual([], sch_failures)
