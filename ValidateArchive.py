import datetime

import FileArchives
import Pass
import Reporter
import Validations


now = datetime.datetime.now()
today = now.strftime('%Y-%m-%d')
reporter = Reporter.CsvReporter('archive-validation-%s.csv' % today)
Pass.PassRunner(reporter).run(FileArchives.get_any_archive(),
                              Validations.std_validation)

# was_converted
