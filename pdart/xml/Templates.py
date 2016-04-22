import xml.dom
import xml.sax


def interpret_text(txt):
    """
    Return a builder function that takes an XML document and returns a
    text node containing the text.
    """
    def builder(doc):
        return doc.createTextNode(txt)
    return builder


def interpret_document_template(template):
    """
    Return a builder function that takes a dictionary and returns an
    XML document containing the template text, with any PARAM elements
    replaced by looking up their 'name' attribute in the dictionary.
    """
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
                        stack.append(elmt)
                    else:
                        elmt = param(doc)
                        if isinstance(elmt, list):
                            for e in elmt:
                                assert isinstance(e, xml.dom.Node)
                        else:
                            assert isinstance(elmt, xml.dom.Node)
                        stack.append(elmt)
                else:
                    elmt = doc.createElement(name)
                    for name in attrs.getNames():
                        elmt.setAttribute(name, attrs[name])
                    stack.append(elmt)

            def endElement(self, name):
                elmt = stack.pop()
                if isinstance(elmt, list):
                    for e in elmt:
                        e.normalize()
                        stack[-1].appendChild(e)
                else:
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


def interpret_template(template):
    """
    Return a parameterizing function that takes a dictionary and
    returns an builder function, performing substitution of the PARAM
    elements with entries from the dictionary, as
    :func:`interpret_document_template` does.

    The returned builder function takes a document and returns XML.
    """
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


def combine_multiple_nodes(doc_funcs):
    """
    Convert a list of functions that take a document and return an XML
    node into a single function that takes a document and returns a
    list of XML nodes.

    [Doc -> Node] -> Doc -> [Node]
    """
    CHECK_TYPES = True
    if CHECK_TYPES:
        assert is_list_of_doc_to_node_functions(doc_funcs)

    def func(document):
        return [doc_func(document) for doc_func in doc_funcs]
    if CHECK_TYPES:
        assert is_doc_to_list_of_nodes_function(func)
    return func


DOC = xml.dom.getDOMImplementation().createDocument(None, None, None)


def is_function(func):
    # a -> b
    return hasattr(func, '__call__')


def is_doc_to_node_function(func):
    # Doc -> Node
    if not is_function(func):
        return False
    return isinstance(func(DOC), xml.dom.Node)


def is_list_of_doc_to_node_functions(func_list):
    # [Doc -> Node]
    if not isinstance(func_list, list):
        return False
    for func in func_list:
        if not is_doc_to_node_function(func):
            return False
    return True


def is_doc_to_list_of_nodes_function(func):
    # Doc -> [Node]
    if not is_function(func):
        return False
    res = func(DOC)
    if isinstance(res, list):
        for n in res:
            if not isinstance(n, xml.dom.Node):
                return False
        return True
    else:
        return False
