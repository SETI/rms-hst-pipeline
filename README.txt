==== INSTALLING THE DEVELOPMENT ENVIRONMENT

To do a quick, automated set-up of your environment, you should have
Miniconda for Python 2.7 installed, probably from
http://conda.pydata.org/miniconda.html.  Miniconda installs Python
versions, tooling, and libraries into an isolated environment, so it
won't affect anything else on your machine.  Yay!

My code currently just assumes that there's a pds-tools package living
in your home directory.  The assumption is encoded (get it?
*encoded*) in pdart/add_pds_tools.py.  I need parts of the package,
but don't want to duplicate it into my repository.  For now, either
make sure you have a copy there by running

> git clone 'https://github.com/SETI/pds-tools.git'

in your home directory, or hack add_pds_tools.py appropriately.

Once Miniconda is installed, running

> ./set-up-environment

creates a Conda environment called 'pdart' and runs basic operations
to ensure it's all working.

==== BASIC OPERATIONS

> source activate pdart

activates the Conda environment, putting your tools and libraries in
scope.

> ./pep8

checks that all the Python is properly formatted according to PEP8.

> ./test

runs the unit tests.

> ./build-api-docs

builds documentation for the PDART modules.  It will automatically
open the main page in your browser.

==== ARCHIVES

Some of the code assumes the existence of an archive.  Up to now, I
(Eric) have been the sole developer, so I've hard-coded my two archive
locations into pdart.pds4.Archives.  

We'll likely need to find a way to add archives, but for now, just
hack something in (write your own 'get_bobs_archive()') and we can
talk about how to do it cleanly.
