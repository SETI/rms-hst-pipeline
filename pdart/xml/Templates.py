"""
This module lets you build XML from templates.

Templates come in two forms: *fragment templates* which represent part
of an XML documents, and *document templates* which represent a
complete XML document.  Both kinds of templates are simply strings
containing legal XML.

Templates may contain holes to be filled in.  There are two types of
holes: *node holes* which get filled with a single XML element, and
*fragment holes*, which may be filled with any number of XML elements.
Node holes are represented by ``<NODE />`` elements with a unique
``name`` attribute.  Fragment holes are represented by ``<FRAGMENT
/>`` elements with a unique ``name`` attribute.

Templates get turned into builder functions that will build the XML.
:func:`interpret_document_template` takes a document template and
returns a builder function.  The builder function takes a dictionary
whose elements are used to fill in the template's holes.  The name
attribute of each hole is used as a key into the dictionary to find
the hole's contents.  The builder function's result is the desired XML
document.

Holes may be filled with ints, floats, or (possibly Unicode) strings
which are automatically converted to XML text nodes, or with builder
functions that take an XML document and return either a single XML
element (for a node hole) or any number of XML elements (for a
fragment hole).  There is some, but minimal typechecking in the
building process, so the programmer needs to be careful.

:func:`interpret_template` takes a fragment template and returns a
function returning a function that must be evaluated in two steps.
First, you give the result a dictionary containing the contents of the
template's holes.  This yields another function, a builder function
that takes a document and returns a portion of an XML document.  This
second function can be used to fill a hole.

A quick example::

    interpret_template('<foo><NODE name="bar"/></foo>')

is a function that takes a dictionary of hole contents.  When we apply
a dictionary to it::

    f = interpret_template('<foo><NODE name="bar"/></foo>')
    d = {'bar': interpret_text('BAR')}
    # d = {'bar': 'BAR'} would also work
    f(d)

we get a builder function that takes an XML document and returns the
XML node ``<foo>BAR</foo>``.  This function can in turn be a value in
a dictionary used to fill a hole in another template.

(In Python's XML implementation, an XML document object serves not
only as the root of the document itself, but also as a factory object
for the nodes that will become part of the document.  So XML documents
must be built top-down: you need the root/factory to build the parts
it contains.  But we describe templates in a bottom-up way, defining a
template in terms of the fragments that make it up.

To compensate for the fact that these go in opposite directions,
instead of turning fragment templates directly into XML (we don't have
the factory yet!), we turn them into builder functions that given a
factory, build XML, and we compose the builder functions into larger
builder functions. A bit convoluted, but it works.

You *could* build from the top down, but the logic wouldn't be any
better; maybe a little worse.)

I document some types below using Haskell notation: *a -> b* is a
function from *a* to *b* and *[c]* is a list of *c* s.
"""
import xml.dom
import xml.sax

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List
    from xml.dom.minidom import Document, Text
    TemplateDict = Dict[str, Any]
    Node = Any  # should be Text
    Frag = List[Text]
    NodeBuilder = Callable[[Document], Node]
    FragBuilder = Callable[[Document], Frag]
    DocTemplate = Callable[[TemplateDict], Document]
    NodeBuilderTemplate = Callable[[TemplateDict], NodeBuilder]


def interpret_text(txt):
    # type: (str) -> NodeBuilder
    """
    Return a builder function that takes an XML document and returns a
    text node containing the text.
    """
    def builder(doc):
        # type: (Document) -> Node
        return doc.createTextNode(txt)
    return builder


def interpret_document_template(template):
    # type: (str) -> DocTemplate
    """
    Return a builder function that takes a dictionary and returns an
    XML document containing the template text, with any ``<NODE />``
    and ``<FRAGMENT />`` elements replaced by looking up their 'name'
    attribute in the dictionary.  ``<NODE />`` elements must evaluate
    to be XML nodes; ``<FRAGMENT />`` elements must evaluate to be an
    XML fragment (a list of XML nodes).
    """
    def builder(dictionary):
        # type: (TemplateDict) -> Document
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        stack = [doc]

        class Builder(xml.sax.ContentHandler):

            def startElement(self, name, attrs):
                if name == 'NODE':
                    param_name = attrs['name']
                    param = dictionary[param_name]
                    if type(param) in [str, unicode]:
                        elmt = doc.createTextNode(param)
                        assert isinstance(elmt, xml.dom.Node), param_name
                        stack.append(elmt)
                    elif type(param) in [int, float]:
                        elmt = doc.createTextNode(str(param))
                        assert isinstance(elmt, xml.dom.Node), param_name
                        stack.append(elmt)
                    else:
                        assert _is_function(param), \
                            ('%s is type %s; should be function' %
                             (param_name, type(param)))
                        elmt = param(doc)
                        if isinstance(elmt, list):
                            for e in elmt:
                                assert isinstance(e, xml.dom.Node), param_name
                        else:
                            assert isinstance(elmt, xml.dom.Node), param_name
                        stack.append(elmt)
                elif name == 'FRAGMENT':
                    param_name = attrs['name']
                    param = dictionary[param_name]
                    assert _is_function(param), \
                        ('%s is type %s; should be function' %
                         (param_name, type(param)))
                    elmts = param(doc)
                    assert isinstance(elmts, list), param_name
                    for elmt in elmts:
                        assert isinstance(elmt, xml.dom.Node), param_name
                    stack.append(elmts)
                else:
                    elmt = doc.createElement(name)
                    for name in attrs.getNames():
                        elmt.setAttribute(name, attrs[name])
                    stack.append(elmt)

            def endElement(self, name):
                if name == 'FRAGMENT':
                    elmts = stack.pop()
                    assert isinstance(elmts, list)
                    for elmt in elmts:
                        assert isinstance(elmt, xml.dom.Node)
                        elmt.normalize()
                        stack[-1].appendChild(elmt)
                else:
                    elmt = stack.pop()
                    assert isinstance(elmt, xml.dom.Node)
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
    # type: (str) -> NodeBuilderTemplate
    """
    Return a parameterizing function that takes a dictionary and
    returns an builder function, performing substitution of the
    ``<NODE />`` and ``<FRAGMENT />`` elements with entries from the
    dictionary, as :func:`interpret_document_template` does.

    The returned builder function takes a document and returns XML.
    """
    def parameterizer(dictionary):
        # type: (TemplateDict) -> NodeBuilder

        def builder(document):
            # type: (Document) -> Node
            doc = document
            stack = []

            class Builder(xml.sax.ContentHandler):

                def startElement(self, name, attrs):
                    if name == 'NODE':
                        param_name = attrs['name']
                        param = dictionary[param_name]
                        if type(param) in [str, unicode]:
                            elmt = doc.createTextNode(param)
                        elif type(param) in [int, float]:
                            elmt = doc.createTextNode(str(param))
                        else:
                            assert _is_function(param), \
                                ('%s is type %s; should be function' %
                                 (param_name, type(param)))
                            elmt = param(doc)
                        assert isinstance(elmt, xml.dom.Node)
                        stack.append(elmt)
                    elif name == 'FRAGMENT':
                        param_name = attrs['name']
                        param = dictionary[param_name]
                        assert _is_function(param), \
                            ('%s is type %s; should be function' %
                             (param_name, type(param)))
                        elmts = param(doc)
                        assert isinstance(elmts, list)
                        for elmt in elmts:
                            assert isinstance(elmt, xml.dom.Node)
                        stack.append(elmts)
                    else:
                        elmt = doc.createElement(name)
                        for name in attrs.getNames():
                            elmt.setAttribute(name, attrs[name])
                        assert isinstance(elmt, xml.dom.Node)
                        stack.append(elmt)

                def endElement(self, name):
                    if name == 'FRAGMENT':
                        elmts = stack.pop()
                        for elmt in elmts:
                            elmt.normalize()
                            if stack:
                                stack[-1].appendChild(elmt)
                            else:
                                stack.append(elmt)
                    else:
                        elmt = stack.pop()
                        elmt.normalize()
                        if stack:
                            stack[-1].appendChild(elmt)
                        else:
                            stack.append(elmt)

                def characters(self, content):
                    node = doc.createTextNode(content)
                    stack[-1].appendChild(node)

            try:
                xml.sax.parseString(template, Builder())
            except Exception:
                print 'malformed template:', template
                raise
            return stack[-1]
        return builder
    return parameterizer


def combine_nodes_into_fragment(doc_funcs):
    # type: (List[NodeBuilder]) -> FragBuilder
    """
    Convert a list of builder functions that take a document and
    return an XML node into a single builder function that takes a
    document and returns an XML fragment (i.e., list of XML nodes).
    """
    def func(document):
        # type: (Document) -> Frag
        return [doc_func(document) for doc_func in doc_funcs]

    return func


def combine_fragments_into_fragment(doc_funcs):
    # type: (List[FragBuilder]) -> FragBuilder
    """
    Convert a list of builder functions that take a document and
    return an XML fragment (list of nodes) into a single builder
    function that takes a document and returns an XML fragment.
    """
    def func(document):
        # type: (Document) -> Frag
        res = []
        for doc_func in doc_funcs:
            res.extend(doc_func(document))
        return res

    return func


_DOC = xml.dom.getDOMImplementation().createDocument(None,
                                                     None,
                                                     None)  \
                                                     # type: Document
"""
A constant document used as a throw-away argument to builder functions
so we can typecheck their results.
"""


def _is_function(func):
    # type: (Any) -> bool
    """
    Return True iff the argument is a function.  We approximate this
    by looking tor a ``__call__`` attribute.

    expected argument type: a -> b
    """
    return hasattr(func, '__call__')


def _is_doc_to_node_function(func):
    # type: (Any) -> bool
    """
    Return True iff the argument is a builder function that takes an
    XML document and returns an XML node.

    expected argument type: Doc -> Node
    """
    return _is_function(func) and isinstance(func(_DOC), xml.dom.Node)


def _is_doc_to_fragment_function(func):
    # type: (Any) -> bool
    """
    Return True iff the argument is a builder function that takes an
    XML document and returns a list of XML nodes.

    expected argument type: Doc -> [Node]
    """
    if not _is_function(func):
        return False
    res = func(_DOC)
    if isinstance(res, list):
        for n in res:
            if not isinstance(n, xml.dom.Node):
                return False
        return True
    else:
        return False
