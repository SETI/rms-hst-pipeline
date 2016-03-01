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
        """
        Return the text appearing at XPath
        '/Product_Observational/Identification_Area/title'.
        """
        return self.PLACEHOLDER('title')

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
        '/Product_Observational/Observation_Area/Investigation_Area/name'.
        """
        return self.PLACEHOLDER('investigation_area_name')

    def investigation_area_type(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/type'.
        """
        return self.CHEATING_PLACEHOLDER('Mission', 'investigation_area_type')

    def internal_reference_type(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/Internal_Reference/reference_type'.
        """
        return 'data_to_investigation'

    def observing_system_component_name(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component/name'.
        """
        return self.PLACEHOLDER('observing_system_component_name')

    def observing_system_component_type(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component/type'.
        """
        return self.CHEATING_PLACEHOLDER('Instrument',
                                         'observing_system_component_type')
