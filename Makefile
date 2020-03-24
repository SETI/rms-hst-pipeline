ACTIVATE=source venv/bin/activate
PIP=python3 -m pip

############################################################
# MYPY TYPECHECKING
############################################################

# Run mypy.

MYPY-FLAGS=--disallow-any-generics \
	--disallow-any-unimported \
	--disallow-untyped-calls \
	--disallow-untyped-defs \
	--strict-equality \
	--warn-redundant-casts \
	--warn-return-any \
	--warn-unreachable

.PHONY: mypy
mypy : venv
	$(ACTIVATE) && MYPYPATH=stubs mypy pdart

############################################################
# TESTS
############################################################

# Run the tests.
.PHONY: test
test: venv
	$(ACTIVATE) && pytest pdart

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

# Format the Python source with black: https://black.readthedocs.io/
.PHONY: black
black :
	black .

# Remove cruft.
.PHONY: tidy
tidy : black
	find . -name '*~' -delete
	find . -name '#*' -delete
	find . -name '*.pyc' -delete

# Remove the virtual environment and cruft.
.PHONY: clean
clean : tidy
	-rm -rf venv
