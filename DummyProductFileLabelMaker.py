import ProductFileLabelMaker


class DummyProductFileLabelMaker(ProductFileLabelMaker.ProductFileLabelMaker):
    """
    A dummy class that builds the part of a PDS4 product label
    corresponding to a single file within the product.  It incorrectly
    declares that the data consists of an array of a single byte.
    This is meant to be used as a placeholder.
    """

    def __init__(self, document, root, archiveFile):
        super(DummyProductFileLabelMaker, self).__init__(document,
                                                         root,
                                                         archiveFile)

    def createFileDataXml(self, fileAreaObservational):
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
