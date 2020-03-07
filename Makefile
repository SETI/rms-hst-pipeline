.PHONY : aq clean java-requirement mtb mypy raw-mypy  save-reqs tar

PYPATH="$(HOME)/fs-copy-on-write:$(HOME)/fs-multiversioned"

# test: I should also run mypy
test : venv java-requirement
	PYTHONPATH=$(PYPATH) source venv/bin/activate && py.test pdart

smalltest : venv java-requirement
	PYTHONPATH=$(PYPATH) source venv/bin/activate && py.test pdart/fs pdart/pipeline

aq : venv
	source venv/bin/activate && python AQ.py

mtb : venv
	source venv/bin/activate && \
	    python MakeTarball.py /Users/spaceman/pdart/new-bulk-download . 11187

# PROJ_IDS=7240 9296 15419
PROJ_IDS=15419
STEPS=copy_primary_files record_changes insert_changes populate_database \
    build_browse build_labels # copy_downloads make_new_versions make_browse

# STEPS=download_docs check_downloads copy_primary_files record_changes # copy_downloads make_new_versions make_browse

pipeline : venv
	# i WFC3 hst_15419 is 101.5MB
	# j ACS hst_09296 is 247.9MB
	# u WFPC2 hst_07240 is 19.8MB
	-rm -rf tmp-working-dir
	tar -xf tmp-working-dir.tar
	for project_id in $(PROJ_IDS); do \
	    for step in $(STEPS); do \
		echo $$project_id $$step; \
		source venv/bin/activate && \
		    PYTHONPATH=$(PYPATH) \
			python Pipeline.py $$project_id $$step; \
            done; \
        done; \
	say I got your pipeline right here
	open tmp-working-dir

tar : tmp-working-dir.tar
	# pass

TAR_STEPS=download_docs check_downloads
tmp-working-dir.tar :
	-rm -rf tmp-working-dir/*
	for project_id in $(PROJ_IDS); do \
	    for step in $(TAR_STEPS); do \
		source venv/bin/activate && \
		    PYTHONPATH=$(PYPATH) \
			python Pipeline.py $$project_id $$step; \
	    done; \
        done; \
	say okay
	tar -cf tmp-working-dir.tar tmp-working-dir


java-requirement :
	@if ! [ -x "$(shell command -v java)" ]; then \
	    echo "**** Java must be installed to run PDART tests ****" ; \
	    exit 1; \
	fi

bulk : venv
	source venv/bin/activate && (python BulkDownload.py | tee bulk.log)

venv : requirements.txt
	virtualenv --no-site-packages -p python2.7 $@
	# pyfits requires numpy to be installed first; dunno why
	source venv/bin/activate && pip install numpy==1.16.4
	source venv/bin/activate && pip install -r requirements.txt

save-reqs :
	source venv/bin/activate && pip freeze > requirements.txt

#
# To use/run mypy
#
mypy : mypy-venv
	source mypy-venv/bin/activate && \
	    PYTHONPATH="$(HOME)/fs-copy-on-write/cowfs" \
                mypy --py2 pdart Pipeline.py | \
	grep -v julian | \
	grep -v picmaker | \
	grep -v "#missing-imports" | \
	grep -v "No library stub file for standard library module 'xml.dom" |\
	grep -v "No library stub file for module 'numpy" | \
	grep -v "No library stub file for module 'sqlalchemy" | \
	grep -v 'Node? has no attribute ' | \
	grep -v "Cannot find module named 'astropy" | \
	grep -v "Cannot find module named 'astroquery" | \
	grep -v "Cannot find module named 'hypothesis" | \
	grep -v "Cannot find module named 'pyfits'" | \
	grep -v "Cannot find module named 'multiversioned." | \
	grep -v "Cannot find module named 'cowfs." | \
	grep -v "Cannot find module named 'pdart.pds4.Archive'" | \
	grep -v "Name 'xml.sax.ContentHandler' is not defined"

raw-mypy : mypy-venv
	source mypy-venv/bin/activate && mypy --py2 pdart Citation_Information

mypy-venv : requirements-mypy.txt
	virtualenv --no-site-packages -p python3 $@
	source mypy-venv/bin/activate && pip install -r requirements-mypy.txt

clean :
	find . -name '*~' -delete
	find . -name '#*' -delete
	find . -name '*.pyc' -delete
	-rm -rf .hypothesis
	-rm -rf apidocs dist MANIFEST
	-rm -rf venv
	-rm -rf mypy-venv
