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
        The text appearing at XPath
        '/Product_Bundle/Identification/Area/title'.
        """
        prodId = str(self.bundle.proposalId())
        return ('This collection contains raw images ' +
                'obtained from HST Observing Program %s.') % prodId

    def citationInformationDescription(self):
        """
        The text appearing at XPath
        '/Product_Bundle/Identification_Area/CitationInformation/description'.
        """
        return self.PLACEHOLDER('citationInformationDescription')

    def citationInformationPublicationYear(self):
        """
        The text appearing at XPath
        '/Product_Bundle/Identification_Area/CitationInformation/publication_year'.
        """
        # This won't pass the XML Schema.
        # return self.PLACEHOLDER('citationInformationPublicationYear')

        # TODO But this is wrong.
        return '2000'
