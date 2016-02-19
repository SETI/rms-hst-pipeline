import FileArchives
import LabelMaker
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
        id_area, obs_area, file_area = \
            self._createChildren(root, ['Identification_Area',
                                        'Observation_Area',
                                        'File_Area_Observational'])

        log_id, vers_id, title, info_ver, prod_cls = \
            self._createChildren(id_area, [
                'pds:logical_identifier',
                'pds:version_id',
                'pds:title',
                'pds:information_model_version',
                'pds:product_class'])

        self._setText(log_id, str(product.lid))
        self._setText(vers_id, self.info.versionID())
        self._setText(title, 'TBD')
        self._setText(info_ver, self.info.informationModelVersion())
        self._setText(prod_cls, 'Product_Observational')

        time_coords, invest_area, obs_sys, targ_id = \
            self._createChildren(obs_area, ['Time_Coordinates',
                                            'Investigation_Area',
                                            'Observing_System',
                                            'Target_Identification'])

        self._createChildren(time_coords, ['start_date_time',
                                           'stop_date_time'])

        self._createChildren(invest_area, ['name', 'type',
                                           'Internal_Reference'])

        obs_sys_comp = self._createChild(obs_sys, 'Observing_System_Component')
        self._createChildren(obs_sys_comp, ['name', 'type'])

        self._createChildren(targ_id, ['name', 'type'])

        for file in product.files():
            self.createFileInfo(file_area, file)

    def createFileInfo(self, file_area, f):
        file = self._createChild(file_area, 'File')
        file_name = self._createChild(file, 'file_name')
        self._setText(file_name, f.basename)
        self._createChild(file_area, 'Array')


def testSynthesis():
    a = FileArchives.getAnyArchive()
    for b in a.bundles():
        if b.proposalId() != 0:
            for c in b.collections():
                for p in c.products():
                    lm = ProductLabelMaker(p)
                    lm.printDefaultXml()
                    return

testSynthesis()
