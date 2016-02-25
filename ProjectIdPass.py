import pyfits

import FileArchives
import HstFilename
import Pass
import Validations


def _add_key_pair_to_set_dict(dict, k, v):
    if k not in dict:
        dict[k] = set()
    dict[k].add(v)


class ProjectIdPass(Pass.NullPass):
    def __init__(self):
        # initialize your tables
        self.bundle_proposal_id = None
        self.num_to_char_dict = {}
        self.char_to_num_dict = {}
        super(ProjectIdPass, self).__init__()

    def do_bundle(self, bundle, before):
        if before:
            self.bundle_proposal_id = bundle.proposal_id()
        else:
            self.bundle_proposal_id = None

    def do_product_file(self, file):
        hst_filename = HstFilename.HstFilename(file.basename)
        hst_proposal_id = hst_filename.hst_internal_proposal_id()
        try:
            proposid = str(pyfits.getval(file.full_filepath(), 'PROPOSID'))
        except IOError:
            proposid = 'IOError'
        except KeyError:
            proposid = 'KeyError'
        _add_key_pair_to_set_dict(self.num_to_char_dict,
                                  proposid, hst_proposal_id)
        _add_key_pair_to_set_dict(self.char_to_num_dict,
                                  hst_proposal_id, proposid)

    def dump(self):
        print repr(self.num_to_char_dict)
        print repr(self.char_to_num_dict)


def runPass():
    archive = FileArchives.get_any_archive()
    p = ProjectIdPass()
    Pass.PassRunner().run(archive,
                          Pass.CompositePass([Validations.CountFilesPass(),
                                              p]))
    p.dump()


if __name__ == '__main__':
    runPass()

# was_converted
