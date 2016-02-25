import pyfits

import FileArchives
import HstFilename
import Pass
import Validations


def _addKeyPairToSetDict(dict, k, v):
    if k not in dict:
        dict[k] = set()
    dict[k].add(v)


class ProjectIdPass(Pass.NullPass):
    def __init__(self):
        # initialize your tables
        self.bundleProposalId = None
        self.numToCharDict = {}
        self.charToNumDict = {}
        super(ProjectIdPass, self).__init__()

    def doBundle(self, bundle, before):
        if before:
            self.bundleProposalId = bundle.proposalId()
        else:
            self.bundleProposalId = None

    def doProductFile(self, file):
        hstFilename = HstFilename.HstFilename(file.basename)
        hstProposalId = hstFilename.hstInternalProposalId()
        try:
            proposId = str(pyfits.getval(file.full_filepath(), 'PROPOSID'))
        except IOError:
            proposId = 'IOError'
        except KeyError:
            proposId = 'KeyError'
        _addKeyPairToSetDict(self.numToCharDict, proposId, hstProposalId)
        _addKeyPairToSetDict(self.charToNumDict, hstProposalId, proposId)

    def dump(self):
        print repr(self.numToCharDict)
        print repr(self.charToNumDict)


def runPass():
    archive = FileArchives.getAnyArchive()
    p = ProjectIdPass()
    Pass.PassRunner().run(archive,
                          Pass.CompositePass([Validations.CountFilesPass(),
                                              p]))
    p.dump()


if __name__ == '__main__':
    runPass()
