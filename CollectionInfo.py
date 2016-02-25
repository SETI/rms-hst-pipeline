import Collection
import Info


class CollectionInfo(Info.Info):
    """
    A helper object associated with a PDS4 Collection providing
    information to fill in collection label fields.
    """

    def __init__(self, collection):
        """Create an object associated with the given Collection."""
        assert isinstance(collection, Collection.Collection)
        self.collection = collection

    def title(self):
        """
        Return the text appearing at XPath
        '/Product_Collection/Identification_Area/title'.
        """
        imageTag = self.collection.suffix().upper() + ' images'
        prodId = str(self.collection.bundle().proposal_id())
        return ('This collection contains the raw %s ' +
                'obtained from HST Observing Program %s.') % (imageTag, prodId)

    def citationInformationDescription(self):
        """
        Return the text appearing at XPath
        '/Product_Collection/Identification_Area/CitationInformation/description'.
        """
        return self.PLACEHOLDER('citationInformationDescription')

    def citationInformationPublicationYear(self):
        """
        Return the text appearing at XPath
        '/Product_Collection/Identification_Area/CitationInformation/publication_year'.
        """
        return self.CHEATING_PLACEHOLDER('2000',
                                         'citationInformationPublicationYear')
