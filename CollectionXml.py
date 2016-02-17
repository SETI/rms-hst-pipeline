import io
import os
import xml.dom

import FileArchives


def createDefaultCollectionInventory(c):
    lines = [u'P,%s\n' % str(p.lid) for p in c.products()]
    # Line endings are native (i.e., wrong) here, but write using
    # io.open() with newline='\r\n'
    return ''.join(lines)


def defaultCollectionInventoryName(c):
    return 'collection_%s_inventory.tab' % c.suffix()


def defaultCollectionXmlName(c):
    return 'collection_%s.xml' % c.suffix()


def createDefaultCollectionXml(collection):
    impl = xml.dom.getDOMImplementation()
    d = impl.createDocument(None, None, None)

    def addProcessingInstruction(lhs, rhs):
        return d.appendChild(d.createProcessingInstruction(lhs, rhs))

    def createChild(parent, name):
        return parent.appendChild(d.createElement(name))

    def createChildren(parent, names):
        return [createChild(parent, name) for name in names]

    def setText(parent, txt):
        return parent.appendChild(d.createTextNode(txt))

    root = createChild(d, 'Product_Collection')

    PDS = 'http://pds.nasa.gov/pds4/pds/v1'
    root.setAttribute('xmlns', PDS)
    root.setAttribute('xmlns:pds', PDS)

    id_area, coll, fa_inv = createChildren(root, ['Identification_Area',
                                                  'Collection',
                                                  'File_Area_Inventory'])

    log_id, vers_id, title, info_ver, prod_cls = createChildren(id_area, [
            'pds:logical_identifier',
            'pds:version_id',
            'pds:title',
            'pds:information_model_version',
            'pds:product_class'])

    setText(log_id, str(collection.lid))
    setText(vers_id, '1.0')
    setText(title, 'TBD')
    setText(info_ver, '1.5.0.0')
    setText(prod_cls, 'Product_Observational')

    setText(createChild(coll, 'collection_type'), 'Data')

    f, inv = createChildren(fa_inv, ['File', 'Inventory'])
    fn = createChild(f, 'file_name')
    setText(fn, defaultCollectionInventoryName(collection))

    off, pars, records, r_delim, \
        f_delim, rec_delim, ref_ty = createChildren(inv, [
            'offset', 'parsing_standard_id',
            'records', 'record_delimiter', 'field_delimiter',
            'Record_Delimited', 'reference_type'])

    setText(off, '0')
    off.setAttribute('unit', 'byte')
    setText(pars, 'PDS DSV 1')

    productCount = 0
    for p in collection.products():
        productCount += 1
    setText(records, str(productCount))

    setText(r_delim, 'carriage-return line-feed')
    setText(f_delim, 'comma')

    fs, gs, f1, f2 = createChildren(rec_delim, [
            'fields', 'groups', 'Field_Delimited', 'Field_Delimited'])
    setText(fs, '2')
    setText(gs, '0')

    nm, fn, dt = createChildren(f1, ['name', 'field_number', 'data_type'])
    setText(nm, 'Member_Status')
    setText(fn, '1')
    setText(dt, 'ASCII_String')

    nm, fn, dt = createChildren(f2, ['name', 'field_number', 'data_type'])
    setText(nm, 'LIDVID_LID')
    setText(fn, '2')
    setText(dt, 'ASCII_LIDVID_LID')

    setText(ref_ty, 'inventory_has_member_product')

    if True:
        return d.toprettyxml(indent='  ', newl='\n', encoding='utf-8')
    else:
        return d.toxml(encoding='utf-8')


def schemaCheck(filepath):
    exitCode = os.system('xmllint --noout --schema "%s" %s' %
                         ('./PDS4_PDS_1500.xsd.xml', filepath))
    return exitCode == 0


def testSynthesis():
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposalId() != 0:
            for c in b.collections():
                xmlName = defaultCollectionXmlName(c)
                xmlSrc = createDefaultCollectionXml(c)
                with open(xmlName, 'w') as f:
                    f.write(xmlSrc)
                invName = defaultCollectionInventoryName(c)
                inventorySrc = createDefaultCollectionInventory(c)
                with io.open(invName, 'w', newline='\r\n') as f:
                    f.write(inventorySrc)
                # print inventorySrc

                if schemaCheck(xmlName):
                    print 'Yay: %s for %s conforms to the schema.' % \
                        (xmlName, str(b))
                else:
                    print ('Boo: %s for %s ' +
                           'does not conform to the schema.') % \
                           (xmlName, str(b))


testSynthesis()
