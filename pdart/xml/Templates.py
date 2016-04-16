import xml.dom
import xml.sax


def interpretText(txt):
    def builder(doc):
        return doc.createTextNode(txt)
    return builder


def interpretDocumentTemplate(template):
    def builder(dictionary):
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        stack = [doc]

        class Builder(xml.sax.ContentHandler):

            def startElement(self, name, attrs):
                if name == 'PARAM':
                    param = dictionary[attrs['name']]
                    if type(param) == str:
                        elmt = doc.createTextNode(param)
                        assert isinstance(elmt, xml.dom.Node)
                    else:
                        elmt = param(doc)
                        assert isinstance(elmt, xml.dom.Node)
                    assert isinstance(elmt, xml.dom.Node)
                else:
                    elmt = doc.createElement(name)
                    for name in attrs.getNames():
                        elmt.setAttribute(name, attrs[name])
                stack.append(elmt)

            def endElement(self, name):
                elmt = stack.pop()
                elmt.normalize()
                stack[-1].appendChild(elmt)

            def characters(self, content):
                node = doc.createTextNode(content)
                stack[-1].appendChild(node)

            def ignorableWhitespace(self, content):
                pass

            def processingInstruction(self, target, data):
                pi = doc.createProcessingInstruction(target, data)
                stack[-1].appendChild(pi)

        xml.sax.parseString(template, Builder())
        return doc

    return builder


def interpretTemplate(template):
    def parameterizer(dictionary):
        def builder(document):
            doc = document
            stack = []

            class Builder(xml.sax.ContentHandler):

                def startElement(self, name, attrs):
                    if name == 'PARAM':
                        param = dictionary[attrs['name']]
                        if type(param) == str:
                            elmt = doc.createTextNode(param)
                        else:
                            elmt = param(doc)
                    else:
                        elmt = doc.createElement(name)
                        for name in attrs.getNames():
                            elmt.setAttribute(name, attrs[name])
                    assert isinstance(elmt, xml.dom.Node)
                    stack.append(elmt)

                def endElement(self, name):
                    elmt = stack.pop()
                    elmt.normalize()
                    if stack:
                        stack[-1].appendChild(elmt)
                    else:
                        stack.append(elmt)

                def characters(self, content):
                    node = doc.createTextNode(content)
                    stack[-1].appendChild(node)

            xml.sax.parseString(template, Builder())
            return stack[-1]
        return builder
    return parameterizer

univDocTemplate = interpretDocumentTemplate(
    '<?xml version="1.0" ?><coffin><PARAM name="body"/></coffin>')


def test2():
    body = interpretTemplate('<body/>')({})
    print univDocTemplate({'body': body}).toxml()


def test():
    template2 = """<x>Namche <PARAM name="baz"/>.
I said <PARAM name="baz"/> not BIZARRE.</x>"""

    interpreted2 = interpretTemplate(template2)

    template = """<?xml version="1.0" ?>
<?pi targ dat?>
<foo>AustraliaFightersFighters
<PARAM name="two"/>
<bar drink="rakshi">elephant</bar>
<bar>eagle</bar>
<bar>tiger</bar>
<bar>shark</bar>
</foo>"""
    interpreted = interpretDocumentTemplate(template)
    dictionary = {'two': interpreted2({'baz': 'bazaar'})}
    print interpreted(dictionary).toxml()

if __name__ == '__main__':
    test()
