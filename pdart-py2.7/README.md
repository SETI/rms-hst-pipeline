# Installing the development environment

To do a quick, automated set-up of your environment, you should have
Miniconda for Python 2.7 installed, probably from
http://conda.pydata.org/miniconda.html.  Miniconda installs Python
versions, tooling, and libraries into an isolated environment, so it
won't affect anything else on your machine.  Yay!

My code currently just assumes that there's a pds-tools package living
in your home directory.  The assumption is encoded (get it?
*encoded*) in ```pdart/add_pds_tools.py```.  I need parts of the
package, but don't want to duplicate it into my repository.  For now,
either let my setup script install it for you, or make sure you have a
copy there by running

```
$ git clone 'https://github.com/SETI/pds-tools.git'
```

in your home directory, or hack ```add_pds_tools.py``` appropriately.

Once Miniconda is installed, running

```
$ ./set-up-environment
```

creates a Conda environment called ```pdart``` and runs basic
operations to ensure it's all working.  Be agreeable and say "yes" to
it repeatedly.

# Basic operations

```
$ source activate pdart
```

activates the Conda environment, putting your tools and libraries in
scope.

```
$ ./pep8
```

checks that all the Python is properly formatted according to PEP8.

```
$ ./test
```

runs the unit tests.

```
$ ./build-api-docs
```

builds documentation for the PDART modules.  It will automatically
open the main page in your browser.

# Archives

Some of the code assumes the existence of an archive (i.e., a
directory packed full of Hubble imagery organized in a certain way).
Up to now, I (Eric) have been the sole developer, so I've just
hard-coded two archive locations into ```pdart.pds4.Archives```.

My big archive lives on the multi-terabyte hard disk on my desk.  My
"mini-archive" is a tiny subset of the big one and it lives on my
development machine.

We'll likely need to find a way to add new archive locations.  For
now, you can just hack something in (write your own
```get_bobs_archive()```) and we can talk later about how to do it
cleanly.

Talk to me if you need to set up an archive.
