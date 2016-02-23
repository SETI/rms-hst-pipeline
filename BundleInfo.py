import Bundle
import Info


class BundleInfo(Info.Info):
    """
    A helper object associated with a PDS4 Bundle providing
    information to fill in bundle label fields.
    """

    def __init__(self, bundle):
        """Create an object associated with the given Bundle."""
        assert isinstance(bundle, Bundle.Bundle)
        self.bundle = bundle

    def title(self):
        """
        Return the text appearing at XPath
        '/Product_Bundle/Identification/Area/title'.
        """
        prodId = str(self.bundle.proposalId())
        return ('This collection contains raw images ' +
                'obtained from HST Observing Program %s.') % prodId

    def citationInformationDescription(self):
        """
        Return the text appearing at XPath
        '/Product_Bundle/Identification_Area/CitationInformation/description'.
        """
        return self.PLACEHOLDER('citationInformationDescription')

    def citationInformationPublicationYear(self):
        """
        Return the text appearing at XPath
        '/Product_Bundle/Identification_Area/CitationInformation/publication_year'.
        """
        return self.CHEATING_PLACEHOLDER('2000',
                                         'citationInformationPublicationYear')
