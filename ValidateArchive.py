import FileArchives
import Pass
import Validations


Pass.PassRunner().run(FileArchives.getAnyArchive(),
                      Validations.stdValidation)
