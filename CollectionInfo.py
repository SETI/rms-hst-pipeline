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
        image_tag = self.collection.suffix().upper() + ' images'
        prod_id = str(self.collection.bundle().proposal_id())
        template = 'This collection contains the raw %s ' + \
            'obtained from HST Observing Program %s.'
        return template % (image_tag, prod_id)

    def citation_information_description(self):
        """
        Return the text appearing at XPath
        '/Product_Collection/Identification_Area/CitationInformation/description'.
        """
        return self.PLACEHOLDER('citation_information_description')

    def citation_information_publication_year(self):
        """
        Return the text appearing at XPath
        '/Product_Collection/Identification_Area/CitationInformation/publication_year'.
        """
        return self.CHEATING_PLACEHOLDER(
            '2000', 'citation_information_publication_year')
