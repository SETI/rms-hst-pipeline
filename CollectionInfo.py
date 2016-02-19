import Collection
import Info


class CollectionInfo(Info.Info):
    def __init__(self, collection):
        assert isinstance(collection, Collection.Collection)
        self.collection = collection

    def title(self):
        return self.PLACEHOLDER('title')
