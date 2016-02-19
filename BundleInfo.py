import Bundle
import Info


class BundleInfo(Info.Info):
    def __init__(self, bundle):
        assert isinstance(bundle, Bundle.Bundle)
        self.bundle = bundle

    def title(self):
        return 'TBD'
