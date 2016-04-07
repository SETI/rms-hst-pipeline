# Experimentation on building XML from text templates
import xml.dom
import xml.sax


def echo_xml(template, params):
    xml.sax.parseString(template.strip(), Echo())


def build_xml(document, template, params):
    builder = Builder(document)
    xml.sax.parseString(template.strip(), builder)


class Echo(xml.sax.ContentHandler):
    def startDocument(self):
        print "startDocument"

    def endDocument(self):
        print "endDocument"

    def startElement(self, name, attrs):
        print "startElement(%s, %s)" % (name, attrs)

    def endElement(self, name):
        print "endElement(%s)" % (name,)

    def characters(self, str):
        print "characters(%r)" % (str,)

    def processingInstruction(self, target, data):
        print "PI(%s, %s)" % (target, data)


class Stack(object):
    def __init__(self):
        self.data = []

    def push_scope(self):
        self.data.append([])

    def pop_scope(self):
        return self.data.pop()

    def push_element(self, elmt):
        self.data[-1].append(elmt)

    def __str__(self):
        return str(self.data)


class Builder(xml.sax.ContentHandler):
    def __init__(self, doc):
        if doc is None:
            doc = xml.dom.getDOMImplementation().createDocument(None,
                                                                None,
                                                                None)
        self.doc = doc
        self.stack = Stack()

    def startDocument(self):
        self.stack.push_scope()

    def endDocument(self):
        scope = self.stack.pop_scope()
        # if you have PIs, len(scope) > 1
        for node in scope:
            print node.toprettyxml()

    def startElement(self, name, attrs):
        self.stack.push_scope()
        self.stack.push_element(name)
        self.stack.push_element(attrs.copy())
        # ignoring attrs for now

    def startElementNS(self, name, qname, attrs):
        self.stack.push_scope()
        self.stack.push_element(name)
        self.stack.push_element(attrs.copy())

    def endElement(self, name):
        self.stack.push_element(name)

        scope = self.stack.pop_scope()
        open_tag = scope[0]
        attrs = scope[1]
        children = scope[2:-1]
        close_tag = scope[-1]
        assert open_tag == close_tag

        if open_tag == 'PARAM':
            elmt = params[attrs['name']]
            if type(elmt) == str:
                elmt = self.doc.createTextNode(elmt)
        else:
            elmt = self.doc.createElement(scope[0])
            for name in attrs.getNames():
                elmt.setAttribute(name, attrs[name])
            for child in children:
                elmt.appendChild(child)
            elmt.normalize()
        self.stack.push_element(elmt)

    def endElementNS(self, name, qname):
        self.stack.push_element(name)
        self.stack.push_element(qname)

        scope = self.stack.pop_scope()
        open_tag = scope[0]
        attrs = scope[1]
        children = scope[2:-1]
        close_tag = scope[-1]
        assert open_tag == close_tag

        if open_tag == 'PARAM':
            elmt = params[attrs['name']]
            if type(elmt) == str:
                elmt = self.doc.createTextNode(elmt)
        else:
            elmt = self.doc.createElement(scope[0])
            for name in attrs.getNames():
                elmt.setAttribute(name, attrs[name])
            for child in children:
                elmt.appendChild(child)
            elmt.normalize()
        self.stack.push_element(elmt)

    def characters(self, content):
        node = self.doc.createTextNode(content)
        self.stack.push_element(node)

    # These have to be handled differently, as they attach to
    # documents, not fragments.  (Not impossible, just not right now.)
    def processingInstruction(self, target, data):
        pi = self.doc.createProcessingInstruction(target, data)
        self.stack.push_element(pi)


if __name__ == '__main__':
    template = """
<?xml version="1.0" ?>
<?xml2 version ?>
<bargaun:foo name="ram bahadur gurung">
bar baz <PARAM name="foo"/></bargaun:foo>"""
    params = {'foo': "FOOBAR!"}
    echo_xml(template, params)
    build_xml(None, template, params)

# This is a prototype.  Need to work out the details for documents
# instead of XML fragments.  And namespaced elements don't seem to be
# working right.
