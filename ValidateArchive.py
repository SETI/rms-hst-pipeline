import FileArchives
import Pass
import Reporter
import Validations


Pass.PassRunner().run(FileArchives.getAnyArchive(),
                      Validations.stdValidation)
