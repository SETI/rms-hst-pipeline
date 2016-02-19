import Product
import Info


class ProductInfo(Info.Info):
    def __init__(self, product):
        assert isinstance(product, Product.Product)
        self.product = product

    def title(self):
        return self.PLACEHOLDER('title')

    def startDateTime(self):
        return self.PLACEHOLDER('startDateTime')

    def stopDateTime(self):
        return self.PLACEHOLDER('stopDateTime')

    def investigationAreaName(self):
        return self.PLACEHOLDER('investigationAreaName')

    def investigationAreaType(self):
        return self.PLACEHOLDER('investigationAreaType')

    def observingSystemComponentName(self):
        return self.PLACEHOLDER('observingSystemComponentName')

    def observingSystemComponentType(self):
        return self.PLACEHOLDER('observingSystemComponentType')

    def targetIdentificationName(self):
        return self.PLACEHOLDER('targetIdentificationName')

    def targetIdentificationType(self):
        return self.PLACEHOLDER('targetIdentificationType')
