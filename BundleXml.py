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
            'logical_identifier',
            'version_id',
            'title',
            'information_model_version',
            'product_class'])

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
            'lid_reference',
            'member_status',
            'reference_type'])
    setText(lid, str(collection.lid))
    setText(stat, 'Primary')
    setText(ref_ty, 'bundle_has_data_collection')

    if True:
        return d.toprettyxml(indent='  ', newl='\n', encoding='utf-8')
    else:
        return d.toxml(encoding='utf-8')

# Print a sample bundle.xml for the first non-hst_00000 bundle.
a = FileArchives.getAnyArchive()
for b in a.bundles():
    if b.proposalId() != 0:
        print createDefaultBundleXml(b)
        break
