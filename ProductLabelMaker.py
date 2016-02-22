import tempfile

import ArchiveFile
import FileArchives
import LabelMaker
import ProductFileInfo
import ProductInfo


class ProductLabelMaker(LabelMaker.LabelMaker):
    def __init__(self, product):
        LabelMaker.LabelMaker.__init__(
            self, product, ProductInfo.ProductInfo(product))

    def defaultXmlName(self):
        assert False, 'ProductLabelMaker.defaultXmlName unimplemented'

    def createDefaultXml(self):
        product = self.component

        root = self._createChild(self.document, 'Product_Observational')

        PDS = self.info.pdsNamespaceUrl()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        id_area, obs_area = \
            self._createChildren(root, ['Identification_Area',
                                        'Observation_Area'])

        log_id, vers_id, title, info_ver, prod_cls = \
            self._createChildren(id_area, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class'])

        self._setText(log_id, str(product.lid))
        self._setText(vers_id, self.info.versionID())
        self._setText(title, self.info.title())
        self._setText(info_ver, self.info.informationModelVersion())
        self._setText(prod_cls, 'Product_Observational')

        time_coords, invest_area, obs_sys, targ_id = \
            self._createChildren(obs_area, ['Time_Coordinates',
                                            'Investigation_Area',
                                            'Observing_System',
                                            'Target_Identification'])

        start_dt, stop_dt = \
            self._createChildren(time_coords, ['start_date_time',
                                               'stop_date_time'])
        self._setText(start_dt, self.info.startDateTime())
        self._setText(stop_dt, self.info.stopDateTime())

        nm, ty, i_ref = \
            self._createChildren(invest_area, ['name', 'type',
                                               'Internal_Reference'])
        self._setText(nm, self.info.investigationAreaName())
        self._setText(ty, self.info.investigationAreaType())

        ref_ty, = self._createChildren(i_ref, ['reference_type'])
        self._setText(ref_ty, self.info.internalReferenceType())

        obs_sys_comp = self._createChild(obs_sys, 'Observing_System_Component')
        nm, ty = self._createChildren(obs_sys_comp, ['name', 'type'])
        self._setText(nm, self.info.observingSystemComponentName())
        self._setText(ty, self.info.observingSystemComponentType())

        nm, ty = self._createChildren(targ_id, ['name', 'type'])
        self._setText(nm, self.info.targetIdentificationName())
        self._setText(ty, self.info.targetIdentificationType())

        for file in product.files():
            self.createFileInfo(root, file)

    def createFileInfo(self, root, f):
        assert isinstance(f, ArchiveFile.ArchiveFile)
        file_area = self._createChild(root, 'File_Area_Observational')

        fileInfo = ProductFileInfo.ProductFileInfo(file_area, f)
        file = self._createChild(file_area, 'File')
        file_name = self._createChild(file, 'file_name')
        self._setText(file_name, fileInfo.fileName())

        # TODO These are the wrong contents; it's a placeholder.
        arr = self._createChild(file_area, 'Array')
        offset, axes, ax_ind_order, elmt_arr, axis_arr = \
            self._createChildren(arr, ['offset', 'axes', 'axis_index_order',
                                       'Element_Array', 'Axis_Array'])

        offset.setAttribute('unit', 'byte')
        self._setText(offset, '0')
        self._setText(axes, '1')
        self._setText(ax_ind_order, 'Last Index Fastest')  # TODO Abstract?

        d_ty = self._createChild(elmt_arr, 'data_type')
        self._setText(d_ty, 'UnsignedByte')

        ax_nm, elmts, seq_num = \
            self._createChildren(axis_arr, ['axis_name', 'elements',
                                            'sequence_number'])

        self._setText(ax_nm, 'kurt')  # TODO Wrong
        self._setText(elmts, '1')  # TODO Wrong
        self._setText(seq_num, '1')  # TODO Wrong


def testSynthesis():
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposalId() != 0:
            for c in b.collections():
                for p in c.products():
                    fp = '/tmp/foo.xml'
                    lm = ProductLabelMaker(p)
                    lm.createDefaultXmlFile(fp)
                    if not (LabelMaker.xmlSchemaCheck(fp) and
                            LabelMaker.schematronCheck(fp)):
                        return

testSynthesis()
