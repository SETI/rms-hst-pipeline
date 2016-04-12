import pdart.pds4.Bundle
import Info


class BundleInfo(Info.Info):
    """
    A helper object associated with a PDS4 Bundle providing
    information to fill in bundle label fields.
    """

    def __init__(self, bundle):
        """Create an object associated with the given Bundle."""
        assert isinstance(bundle, pdart.pds4.Bundle.Bundle)
        self.bundle = bundle

    def title(self):
        """
        Return the text appearing at XPath
        '/Product_Bundle/Identification/Area/title'.
        """
        prod_id = str(self.bundle.proposal_id())
        return ('This bundle contains images ' +
                'obtained from HST Observing Program %s.') % prod_id

    def citation_information_description(self):
        """
        Return the text appearing at XPath
        '/Product_Bundle/Identification_Area/CitationInformation/description'.
        """
        return self.PLACEHOLDER('citation_information_description')

    def citation_information_publication_year(self):
        """
        Return the text appearing at XPath
        '/Product_Bundle/Identification_Area/CitationInformation/publication_year'.
        """
        return self.CHEATING_PLACEHOLDER(
            '2000', 'citation_information_publication_year')
