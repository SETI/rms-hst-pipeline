import FileArchives
import Validation
import Validations


Validation.runArchiveValidation(FileArchives.getAnyArchive(),
                                Validations.stdValidation)
