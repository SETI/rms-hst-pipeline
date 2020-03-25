import unittest
import xml.dom

from pdart.xml.Templates import (
    interpret_document_template,
    interpret_template,
    interpret_text,
)


class TestTemplates(unittest.TestCase):
    def test_interpret_text(self) -> None:
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        txt = interpret_text("foo")(doc)
        self.assertTrue("foo", txt.data)

    def test_interpret_document_template(self) -> None:
        body = interpret_document_template("<body/>")
        self.assertEqual('<?xml version="1.0" ?><body/>', body({}).toxml())

        body_with_param = interpret_document_template(
            '<body_with_param><NODE name="foo"/></body_with_param>'
        )
        self.assertEqual(
            '<?xml version="1.0" ?><body_with_param>bar</body_with_param>',
            body_with_param({"foo": "bar"}).toxml(),
        )

    def test_interpret_template(self) -> None:
        make_body = interpret_document_template('<doc><NODE name="foo"/></doc>')

        make_template = interpret_template('<foo><NODE name="bar"/></foo>')

        template = make_template({"bar": interpret_text("BAR")})
        body = make_body({"foo": template})

        self.assertEqual(
            '<?xml version="1.0" ?><doc><foo>BAR</foo></doc>', body.toxml()
        )
