import os
import xml.dom

import FileArchives


def createDefaultBundleXml(bundle):
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

    root = createChild(d, 'Product_Bundle')

    PDS = 'http://pds.nasa.gov/pds4/pds/v1'
    root.setAttribute('xmlns', PDS)
    root.setAttribute('xmlns:pds', PDS)

    id_area, b = createChildren(root, ['Identification_Area', 'Bundle'])

    log_id, vers_id, title, info_ver, prod_cls = createChildren(id_area, [
            'pds:logical_identifier',
            'pds:version_id',
            'pds:title',
            'pds:information_model_version',
            'pds:product_class'])

    setText(log_id, str(bundle.lid))
    setText(vers_id, '1.0')
    setText(title, 'TBD')
    setText(info_ver, '1.5.0.0')
    setText(prod_cls, 'Product_Observational')

    b_ty = createChild(b, 'bundle_type')
    setText(b_ty, 'Archive')

    for collection in bundle.collections():
        mem_entry = createChild(root, 'Bundle_Member_Entry')
        lid, stat, ref_ty = createChildren(mem_entry, [
                'pds:lid_reference',
                'pds:member_status',
                'pds:reference_type'])
        setText(lid, str(collection.lid))
        setText(stat, 'Primary')
        setText(ref_ty, 'bundle_has_data_collection')

    if True:
        return d.toprettyxml(indent='  ', newl='\n', encoding='utf-8')
    else:
        return d.toxml(encoding='utf-8')


def schemaCheck(filepath):
    exitCode = os.system('xmllint --noout --schema "%s" %s' %
                         ('./PDS4_PDS_1500.xsd.xml', filepath))
    return exitCode == 0

if False:
    # Create sample bundle.xml files for the non-hst_00000 bundles and
    # test them against the XML schema.
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposalId() != 0:
            xmlSrc = createDefaultBundleXml(b)
            with open('bundle.xml', 'w') as f:
                f.write(xmlSrc)
            if schemaCheck('bundle.xml'):
                print 'Yay: bundle.xml for %s conforms to the schema.' % str(b)
            else:
                print ('Boo: bundle.xml for %s ' +
                       'does not conform to the schema.') % str(b)
