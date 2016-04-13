import datetime

import pdart.pds4.Archives
import Pass
import Reporter
import Validations


now = datetime.datetime.now()
today = now.strftime('%Y-%m-%d')
reporter = Reporter.CsvReporter('archive-validation-%s.csv' % today)
Pass.PassRunner(reporter).run(pdart.pds4.Archives.get_any_archive(),
                              Validations.std_validation)
