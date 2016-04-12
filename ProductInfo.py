import pdart.pds4.Product
import Info


class ProductInfo(Info.Info):
    """
    A helper object associated with a PDS4 Product providing
    information to fill in product label fields.
    """

    def __init__(self, product):
        """Create an object associated with the given Product."""
        assert isinstance(product, pdart.pds4.Product.Product)
        self.product = product

    def title(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Identification_Area/title'.
        """
        # TODO This needs to be updated since making products contain
        # only one image.

        collection = self.product.collection()
        image_tag = collection.suffix().upper() + ' image'
        # visit = self.product.visit()
        prod_id = str(collection.bundle().proposal_id())
        template = 'This collection contains the %s ' + \
            'obtained the HST Observing Program %s.'
        return template % (image_tag, prod_id)

    def start_date_time(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Time_Coordinates/start_date_time'.
        """
        return self.CHEATING_PLACEHOLDER('2000-01-02Z', 'start_date_time')

    def stop_date_time(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Time_Coordinates/stop_date_time'.
        """
        return self.CHEATING_PLACEHOLDER('2000-01-02Z', 'stop_date_time')

    def investigation_area_name(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/name'
        """
        return 'HST observing program %d' % self.product.bundle().proposal_id()

    def investigation_area_type(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/type'
        """
        return 'Individual Investigation'

    def investigation_lidvid_reference(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/Internal_Reference/lidvid_reference'
        """
        template = \
            'urn:nasa:pds:context:investigation:investigation.hst_%05d::1.0'
        return template % self.product.bundle().proposal_id()

    def internal_reference_type(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/Internal_Reference/reference_type'.
        """
        return 'data_to_investigation'

    def modification_history_description(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Identification_Area/Modification_History/description'
        """
        return 'PDS4 version-in-development of the product'
