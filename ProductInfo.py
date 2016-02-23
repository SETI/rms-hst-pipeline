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

    def startDateTime(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Time_Coordinates/start_date_time'.
        """
        return self.CHEATING_PLACEHOLDER('2000-01-02Z', 'startDateTime')

    def stopDateTime(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Time_Coordinates/stop_date_time'.
        """
        return self.CHEATING_PLACEHOLDER('2000-01-02Z', 'stopDateTime')

    def investigationAreaName(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/name'.
        """
        return self.PLACEHOLDER('investigationAreaName')

    def investigationAreaType(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/type'.
        """
        return self.CHEATING_PLACEHOLDER('Mission', 'investigationAreaType')

    def internalReferenceType(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Investigation_Area/Internal_Reference/reference_type'.
        """
        return 'data_to_investigation'

    def observingSystemComponentName(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component/name'.
        """
        return self.PLACEHOLDER('observingSystemComponentName')

    def observingSystemComponentType(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Observing_System/Observing_System_Component/type'.
        """
        return self.CHEATING_PLACEHOLDER('Instrument',
                                         'observingSystemComponentType')

    def targetIdentificationName(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Target_Identification/name'.
        """
        return self.PLACEHOLDER('targetIdentificationName')

    def targetIdentificationType(self):
        """
        Return the text appearing at XPath
        '/Product_Observational/Observation_Area/Target_Identification/type'.
        """
        return self.CHEATING_PLACEHOLDER('Planet', 'targetIdentificationType')
