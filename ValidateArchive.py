import FileArchives
import Pass
import Validations


Pass.runArchivePasses(FileArchives.getAnyArchive(),
                      Validations.stdValidation)
