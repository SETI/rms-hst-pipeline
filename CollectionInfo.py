import Collection
import Info


class CollectionInfo(Info.Info):
    def __init__(self, collection):
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
        # TODO This is wrong.
        return '2000'
