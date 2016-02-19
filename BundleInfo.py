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
