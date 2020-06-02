ACTIVATE=source venv/bin/activate
PIP=python3 -m pip

# Format, typecheck, and test.
.PHONY: all
all : black mypy test

############################################################
# MYPY TYPECHECKING
############################################################

# Run mypy.

MYPY_FLAGS=--disallow-any-unimported \
	--disallow-untyped-calls \
	--disallow-untyped-defs \
	--strict-equality \
	--warn-redundant-casts \
	--warn-unreachable \

# --disallow-any-generics: produces baffling errors.  At first, it
# seemed as if, for example, re.Pattern was both generic and not.
# Turns out re.Pattern and typing.Pattern are difference.  But it
# hasn't been worth following this path right now.  Maybe later.

# --warn-return-any: not practical because of SqlAlchemy's dynamic magic
# and because FITS cards are untyped.

.PHONY: mypy
mypy : venv
	$(ACTIVATE) && MYPYPATH=stubs mypy $(MYPY_FLAGS) *.py pdart

############################################################
# TESTS
############################################################

# Run the tests.
.PHONY: test
test: venv
	$(ACTIVATE) && PYTHONPATH=$(HOME)/pds-tools pytest pdart

# Run some subset of the tests.  Hack as needed.
.PHONY: t
t: venv
	$(ACTIVATE) && PYTHONPATH=$(HOME)/pds-tools pytest pdart/pipeline

############################################################
# THE PIPELINE
############################################################

ACS_IDS=09059 09296 09678 09985 10461 10719 10774 11109 13012 13199 15098 10506
WFC3_IDS=11536 12436 13713 13716 13936 14136 14491 15142 15233 15419 15581 13118
WFPC2_IDS=05219 05167 05238 05640 05633 05783 05824 06145 06621 07240 11497 11361

PROJ_IDS=$(ACS_IDS) $(WFC3_IDS) $(WFPC2_IDS)


STEPS=reset_pipeline download_docs check_downloads copy_primary_files	\
    record_changes insert_changes populate_database build_browse	\
    build_labels update_archive make_deliverable validate_bundle

.PHONY: pipeline
pipeline : venv clean-results
	mkdir -p tmp-working-dir
	-rm tmp-working-dir/*/\#*.txt
	-rm tmp-working-dir/*/KEY_ERROR_DUMP.txt
	for project_id in $(PROJ_IDS); do \
	    for step in $(STEPS); do \
		echo $$project_id $$step; \
		$(ACTIVATE) && \
		    PYTHONPATH=$(HOME)/pds-tools \
			python Pipeline.py $$project_id $$step; \
            done; \
        done; \
	say pipeline is done
	open tmp-working-dir

.PHONY: results
results :
	@find tmp-working-dir -name '#*' | sort

.PHONY: clean-results
clean-results :
	find tmp-working-dir -name '#*' -delete

TARGET_DIR=$(HOME)/zips

.PHONY : copy-results
copy-results :
	rm -rf $(TARGET_DIR)/*
	for dir in `find tmp-working-dir -name '*-deliverable'`; do \
	    pushd $$dir; \
	    zip -X -r $(TARGET_DIR)/`basename $$dir`.zip . ;\
	    popd; \
	done

############################################################
# THE VIRTUAL ENVIRONMENT
############################################################

# Install the virtual environment.
venv : requirements.txt
	python3 -m venv venv
	$(ACTIVATE) && \
	    $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

# After you've hand-installed new packages, save them to the
# requirements list.
.PHONY: save-reqs
save-reqs :
	$(ACTIVATE) && $(PIP) freeze > requirements.txt
	touch venv  # to prevent rebuilding

############################################################
# MAINTENANCE
############################################################

# Show a diagram of the module dependencies.  Do not break these.
.PHONY: modules
modules:
	dot -Tpng modules.dot -o modules.png
	open modules.png

# Format the Python source with black: https://black.readthedocs.io/
.PHONY: black
black :
	black .

# Remove cruft.
.PHONY: tidy
tidy : black
	-rm modules.png
	find . -name '*~' -delete
	find . -name '#*' -delete
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

# Remove the virtual environment and cruft.
.PHONY: clean
clean : tidy
	-rm -rf venv
