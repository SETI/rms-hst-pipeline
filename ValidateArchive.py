import FileArchives
import Pass
import Reporter
import Validations


reporter = Reporter.CsvReporter('archive-validation.csv')
Pass.PassRunner(reporter).run(FileArchives.getAnyArchive(),
                              Validations.stdValidation)
