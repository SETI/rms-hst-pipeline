import Product
import Info


class ProductInfo(Info.Info):
    """
    A helper object associated with a PDS4 Product providing
    information to fill in product label fields.
    """

    def __init__(self, product):
        """Create an object associated with the given Product."""
        assert isinstance(product, Product.Product)
        self.product = product

    def title(self):
        return self.PLACEHOLDER('title')

    def startDateTime(self):
        # TODO This is wrong
        return '2000-01-02Z'
        return self.PLACEHOLDER('startDateTime')

    def stopDateTime(self):
        # TODO This is wrong
        return '2000-01-02Z'
        return self.PLACEHOLDER('stopDateTime')

    def investigationAreaName(self):
        return self.PLACEHOLDER('investigationAreaName')

    def investigationAreaType(self):
        # TODO This is wrong
        return 'Mission'
        return self.PLACEHOLDER('investigationAreaType')

    def internalReferenceType(self):
        return 'data_to_investigation'

    def observingSystemComponentName(self):
        return self.PLACEHOLDER('observingSystemComponentName')

    def observingSystemComponentType(self):
        # TODO This is wrong
        return 'Instrument'
        return self.PLACEHOLDER('observingSystemComponentType')

    def targetIdentificationName(self):
        return self.PLACEHOLDER('targetIdentificationName')

    def targetIdentificationType(self):
        # TODO This is wrong
        return 'Planet'
        return self.PLACEHOLDER('targetIdentificationType')
