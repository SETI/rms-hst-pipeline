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
        return self.CHEATING_PLACEHOLDER('2000-01-02Z', 'startDateTime')

    def stopDateTime(self):
        return self.CHEATING_PLACEHOLDER('2000-01-02Z', 'stopDateTime')

    def investigationAreaName(self):
        return self.PLACEHOLDER('investigationAreaName')

    def investigationAreaType(self):
        return self.CHEATING_PLACEHOLDER('Mission', 'investigationAreaType')

    def internalReferenceType(self):
        return 'data_to_investigation'

    def observingSystemComponentName(self):
        return self.PLACEHOLDER('observingSystemComponentName')

    def observingSystemComponentType(self):
        return self.CHEATING_PLACEHOLDER('Instrument',
                                         'observingSystemComponentType')

    def targetIdentificationName(self):
        return self.PLACEHOLDER('targetIdentificationName')

    def targetIdentificationType(self):
        return self.CHEATING_PLACEHOLDER('Planet', 'targetIdentificationType')
