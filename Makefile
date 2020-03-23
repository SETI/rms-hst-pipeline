ACTIVATE=source venv/bin/activate
MYPY-ACTIVATE=source mypy-venv/bin/activate
PIP=python3 -m pip

############################################################
# MYPY TYPECHECKING
############################################################

# Run mypy.
.PHONY: mypy
mypy : mypy-venv
	$(MYPY-ACTIVATE) && mypy pdart

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

#### MAIN ####

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

#### MYPY ####

# Install the mypy virtual environment.
mypy-venv : mypy-requirements.txt
	python3 -m venv mypy-venv
	$(MYPY-ACTIVATE) && \
	    $(PIP) install --upgrade pip && \
	    $(PIP) install -r mypy-requirements.txt

# After you've hand-installed new packages for mypy, save them to the
# requirements list.
.PHONY: mypy-save-reqs
mypy-save-reqs :
	$(MYPY-ACTIVATE) && $(PIP) freeze > mypy-requirements.txt
	touch mypy-venv  # to prevent rebuilding

############################################################
# MAINTENANCE
############################################################

# Format the Python source with black: https://black.readthedocs.io/
.PHONY: black
black :
	black .

# Remove cruft.
.PHONY: tidy
tidy :
	find . -name '*~' -delete
	find . -name '#*' -delete
	find . -name '*.pyc' -delete
	-rm -rf venv

# Remove the virtual environment and cruft.
.PHONY: clean
clean : tidy
	find . -name '*~' -delete
	find . -name '#*' -delete
	find . -name '*.pyc' -delete
	-rm -rf venv mypy-venv
