==== INSTALLING THE DEVELOPMENT ENVIRONMENT

To do a quick, automated set-up of your environment, you should have
Miniconda for Python 2.7 installed, probably from
http://conda.pydata.org/miniconda.html.  Miniconda installs Python
versions, tooling, and libraries into an isolated environment, so it
won't affect anything else on your machine.  Yay!

Once Miniconda is installed, running

> ./set-up-environment

creates a Conda environment called 'pdart' and runs basic operations
to ensure it's all working.

==== BASIC OPERATIONS

> source activate pdart

activates the environment.

> ./pep8

checks that all the Python is properly formatted according to PEP8.

> ./test

runs the unit tests.

> ./build-api-docs

builds documentation for the PDART modules.  It will automatically
open the main page in your browser.


