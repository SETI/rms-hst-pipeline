import Product
import Info


class ProductInfo(Info.Info):
    def __init__(self, product):
        assert isinstance(product, Product.Product)
        self.product = product
