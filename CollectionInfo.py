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
        imageTag = self.collection.suffix().upper() + ' images'
        prodId = str(self.collection.bundle().proposalId())
        return ('This collection contains the raw %s ' +
                'obtained from HST Observing Program %s.') % (imageTag, prodId)

    def citationInformationDescription(self):
        return self.PLACEHOLDER('citationInformationDescription')

    def citationInformationPublicationYear(self):
        return self.CHEATING_PLACEHOLDER('2000',
                                         'citationInformationPublicationYear')
