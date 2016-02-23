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

        root = self.createChild(self.document, 'Product_Observational')

        PDS = self.info.pdsNamespaceUrl()
        root.setAttribute('xmlns', PDS)
        root.setAttribute('xmlns:pds', PDS)

        identificationArea, observationArea = \
            self.createChildren(root, ['Identification_Area',
                                       'Observation_Area'])

        logicalIdentifier, versionId, title, \
            informationModelVersion, productClass = \
            self.createChildren(identificationArea, [
                'logical_identifier',
                'version_id',
                'title',
                'information_model_version',
                'product_class'])

        self.setText(logicalIdentifier, str(product.lid))
        self.setText(versionId, self.info.versionID())
        self.setText(title, self.info.title())
        self.setText(informationModelVersion,
                     self.info.informationModelVersion())
        self.setText(productClass, 'Product_Observational')

        timeCoordinates, investigationArea, \
            observingSystem, targetIdentification = \
            self.createChildren(observationArea, ['Time_Coordinates',
                                                  'Investigation_Area',
                                                  'Observing_System',
                                                  'Target_Identification'])

        startDateTime, stopDateTime = \
            self.createChildren(timeCoordinates, ['start_date_time',
                                                  'stop_date_time'])
        self.setText(startDateTime, self.info.startDateTime())
        self.setText(stopDateTime, self.info.stopDateTime())

        name, type, internalReference = \
            self.createChildren(investigationArea, ['name', 'type',
                                                    'Internal_Reference'])
        self.setText(name, self.info.investigationAreaName())
        self.setText(type, self.info.investigationAreaType())

        referenceType = self.createChild(internalReference, 'reference_type')
        self.setText(referenceType, self.info.internalReferenceType())

        observingSystemComponent = \
            self.createChild(observingSystem, 'Observing_System_Component')
        name, type = self.createChildren(observingSystemComponent,
                                         ['name', 'type'])
        self.setText(name, self.info.observingSystemComponentName())
        self.setText(type, self.info.observingSystemComponentType())

        name, type = self.createChildren(targetIdentification,
                                         ['name', 'type'])
        self.setText(name, self.info.targetIdentificationName())
        self.setText(type, self.info.targetIdentificationType())

        for file in product.files():
            self.createFileInfo(root, file)

    def createFileInfo(self, root, archiveFile):
        assert isinstance(archiveFile, ArchiveFile.ArchiveFile)
        fileAreaObservational = self.createChild(root,
                                                 'File_Area_Observational')

        fileInfo = ProductFileInfo.ProductFileInfo(fileAreaObservational,
                                                   archiveFile)
        file = self.createChild(fileAreaObservational, 'File')
        file_name = self.createChild(file, 'file_name')
        self.setText(file_name, fileInfo.fileName())

        # TODO These are the wrong contents; it's a placeholder.
        array = self.createChild(fileAreaObservational, 'Array')
        offset, axes, axisIndexOrder, elementArray, axisArray = \
            self.createChildren(array, ['offset', 'axes', 'axis_index_order',
                                        'Element_Array', 'Axis_Array'])

        offset.setAttribute('unit', 'byte')
        self.setText(offset, '0')
        self.setText(axes, '1')
        self.setText(axisIndexOrder, 'Last Index Fastest')  # TODO Abstract?

        dataType = self.createChild(elementArray, 'data_type')
        self.setText(dataType, 'UnsignedByte')

        axisName, elements, sequenceNumber = \
            self.createChildren(axisArray, ['axis_name', 'elements',
                                            'sequence_number'])

        self.setText(axisName, 'Axis Joe')  # TODO Wrong
        self.setText(elements, '1')  # TODO Wrong
        self.setText(sequenceNumber, '1')  # TODO Wrong


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
