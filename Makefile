# The Python that gets used within the virtual environment is the
# Python you use to create the virtual environment.  It's usually
# called "python3", but if you need to use a different one, you can
# override it here.
#
# The variable $(PYTHON) only used in the goal "venv" when creating
# the virtual environment. After that, we always activate the virtual
# environment before running Python, so we don't have to specific
# which one.
PYTHON=python3

ACTIVATE=source venv/bin/activate
PIP=python -m pip

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

.PHONY: print-var
print-var:
	@echo $(HOME)
	@echo $(TMP_WORKING_DIR)
	@echo $(PDSTOOLS_PATH)

.PHONY: mypy
mypy : venv
	$(ACTIVATE) && MYPYPATH=stubs mypy $(MYPY_FLAGS) *.py pdart

experiment : venv black mypy
	$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) python Experiment.py


############################################################
# TESTS
############################################################

# Run the tests.
.PHONY: test
test: venv
	$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) pytest pdart

# Run some subset of the tests.  Hack as needed.
.PHONY: t
t: venv
	$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) pytest pdart/labels/test_HstParameters.py


############################################################
# THE PIPELINE
############################################################

TWD=$(TMP_WORKING_DIR)
ZIPS=$(TMP_WORKING_DIR)/zips

ACS_IDS=09059 09296 09440 09678 09725 09745 09746 09985 10192 10461	\
10502 10506 10508 10545 10719 10774 10783 11055 11109 12601 13012	\
13199 13632 13795 15098

WFC3_IDS=11536 11656 12237 12245 12436 12894 13118 13713 13716 13873	\
13936 14044 14045 14113 14136 14491 14928 14939 15142 15143 15233	\
15419 15456 15505 15581

WFPC2_IDS=05167 05219 05220 05238 05493 05508 05633 05640 05783 05824	\
05828 05829 06025 06145 06215 06295 06621 06736 06842 07240 07276	\
11102 11361 11497 11956

PROJ_IDS=$(ACS_IDS) $(WFC3_IDS) $(WFPC2_IDS)
# PROJ_IDS=09296 15098 11536 15581 05167 11956


# STEPS=reset_pipeline download_docs check_downloads copy_primary_files	\
#     record_changes insert_changes populate_database build_browse	\
#     build_labels update_archive make_deliverable validate_bundle

.PHONY: pipeline
pipeline : venv clean-results
	mkdir -p $(TWD)
	-rm $(TWD)/hst_*/\#*.txt
	for project_id in $(PROJ_IDS); do \
	    echo '****' hst_$$project_id '****'; \
	    $(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
		python Pipeline.py $$project_id; \
	done;
	say pipeline is done
	open $(TWD)

.PHONY: results
results :
	@ls $(TWD)/hst_*/\#* | sort

.PHONY: clean-results
clean-results :
	-rm $(TWD)/hst_*/\#*


.PHONY : copy-results
copy-results :
	rm -rf $(ZIPS)/
	mkdir -p $(ZIPS)/
	for dir in `find $(TWD) -name '*-deliverable'`; do \
	    pushd $$dir; \
	    zip -X -r $(ZIPS)/`basename $$dir`.zip . ;\
	    popd; \
	done

##############################
# smaller version for testing
##############################

LILS=07885 09059 09748 15505

.PHONY: lil-pipeline
LIL-TWD=$(TMP_WORKING_DIR)
lil-pipeline : venv
	mkdir -p $(LIL-TWD)
	-rm $(LIL-TWD)/*/\#*.txt
	for project_id in $(LILS); do \
	    echo '****' hst_$$project_id '****'; \
	    $(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
		python Pipeline.py $$project_id; \
	done;
	say lil pipeline is done
	open $(LIL-TWD)

##############################
# Pipeline for NICMOS ONLY
##############################
nicmos-pipeline : LILS=07885
.PHONY: nicmos-pipeline
nicmos-pipeline : lil-pipeline

##############################
# Download shm & spt from mast
##############################
TEST_ID=07885 09059 09748 15505

.PHONY: download-shm-spt
download-shm-spt : setup_dir
	for project_id in $(PROJ_IDS); do \
		echo '****' hst_$$project_id '****'; \
		$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
		python Download_SHM_SPT.py $$project_id; \
	done;

############################################################
# Download shm & spt from mast for all proposal ids with mtflag=True
############################################################
ID_LIST=proposal_ids_all.txt
download-shm-spt-all : get-proposal-ids
download-shm-spt-all : PROJ_IDS=$(shell cat ${ID_LIST})
.PHONY: download-shm-spt-all
download-shm-spt-all : download-shm-spt

############################################################
# Get the list of proposal ids with moving target = true
############################################################
.PHONY: get-proposal-ids
get-proposal-ids : venv
	mkdir -p $(LIL-TWD)
	$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
	python GetProposalIds.py all; \
	echo '**** List of Proposal Ids is created under' $(LIL-TWD) '****'; \

############################################################
# Get the list of proposal ids with image product type & moving target = true
############################################################
.PHONY: get-image-proposal-ids
get-image-proposal-ids : venv
	mkdir -p $(LIL-TWD)
	$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
	python GetProposalIds.py image; \
	echo '**** List of Proposal Ids is created under' $(LIL-TWD) '****'; \

############################################################
# Get file names and unique suffixes
############################################################
get-file-names-suffixes : TEST_IDS=05150 12037 04521 07313 03744 05844
get-file-names-suffixes : FILES_PATH=$(LIL-TWD)/files_from_mast
.PHONY: get-file-names-suffixes
get-file-names-suffixes : venv
	mkdir -p $(LIL-TWD)
	for project_id in $(TEST_IDS); do \
		echo '****' hst_$$project_id '****'; \
		$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
		python DownloadAllFilesForOneProposalId.py $$project_id; \
	done;
	echo '**** List of file names & suffixes is created under' $(FILES_PATH) '****'; \


############################################################
# Get file names and unique suffixes for all proposal ids with mtflag=True
############################################################
ID_LIST=proposal_ids_all.txt
get-file-names-suffixes-all : get-proposal-ids
get-file-names-suffixes-all : TEST_IDS=$(shell cat ${ID_LIST})
get-file-names-suffixes-all : FILES_PATH=$(LIL-TWD)/files_from_mast
.PHONY: get-file-names-suffixes
get-file-names-suffixes-all : venv
	mkdir -p $(LIL-TWD)
	for project_id in $(TEST_IDS); do \
		echo '****' hst_$$project_id '****'; \
		$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
		python DownloadAllFilesForOneProposalId.py $$project_id; \
	done;
	echo '**** List of file names & suffixes is created under' $(FILES_PATH) '****'; \

############################################################
# Download all files for one proposal id
############################################################
download-all-files : TEST_IDS=05150 12037 04521 07313 03744 05844
download-all-files : FILES_PATH=$(LIL-TWD)/files_from_mast
.PHONY: get-file-names-suffixes
download-all-files : venv
	mkdir -p $(LIL-TWD)
	for project_id in $(TEST_IDS); do \
		echo '****' hst_$$project_id '****'; \
		$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) \
		python DownloadAllFilesForOneProposalId.py $$project_id -d; \
	done;
	echo '**** Files are under' $(FILES_PATH) '****'; \

############################################################
# Setup
############################################################
setup_dir : venv
	mkdir -p $(LIL-TWD)
	-rm $(LIL-TWD)/*/\#*.txt

############################################################
# CHECK SUBARRAY FLAG
############################################################

.PHONY : check
check :
	$(ACTIVATE) && (python CheckSubarrayFlag.py | tee check.out)

############################################################
# THE VIRTUAL ENVIRONMENT
############################################################

# Install the virtual environment.
venv : requirements.txt
	$(PYTHON) -m venv venv
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
black : venv
	$(ACTIVATE) && black .

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


.PHONY: target_files
target_files : venv
	$(ACTIVATE) && PYTHONPATH=$(PDSTOOLS_PATH) python DownloadTargetFiles.py
