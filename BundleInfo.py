import Bundle
import Info


class BundleInfo(Info.Info):
    def __init__(self, bundle):
        assert isinstance(bundle, Bundle.Bundle)
        self.bundle = bundle

    def title(self):
        prodId = str(self.bundle.proposalId())
        return ('This collection contains raw images ' +
                'obtained from HST Observing Program %s.') % prodId

    def citationInformationDescription(self):
        return self.PLACEHOLDER('citationInformationDescription')

    def citationInformationPublicationYear(self):
        # This won't pass the XML Schema.
        # return self.PLACEHOLDER('citationInformationPublicationYear')

        # TODO But this is wrong.
        return '2000'
