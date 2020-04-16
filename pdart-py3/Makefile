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

# --disallow-any-generics: produces baffling errors.  It seems as if,
# for example, re.Pattern is both generic and not.  Perhaps a mismatch
# between the source and typing stubs, or a versioning problem.

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

.PHONY: t
t: venv
	$(ACTIVATE) && PYTHONPATH=$(HOME)/pds-tools pytest pdart/fs/multiversioned

############################################################
# THE PIPELINE
############################################################

ACS_IDS=09296 10719 11109 13012
WFC3_IDS=11536 14491 15419 13936
WFPC2_IDS=07240 05783 06621 11497

PROJ_IDS=$(ACS_IDS) $(WFC3_IDS) $(WFPC2_IDS)

STEPS=reset_pipeline download_docs check_downloads copy_primary_files	\
    record_changes insert_changes populate_database build_browse	\
    build_labels update_archive make_deliverable validate_bundle

.PHONY: pipeline
pipeline : venv
	mkdir -p tmp-working-dir
	-rm tmp-working-dir/*/LAST*.txt  # can't say dollar-sign
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
