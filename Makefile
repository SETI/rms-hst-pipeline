.PHONY : aq clean java-requirement mtb mypy raw-mypy  save-reqs tar


# test: I should also run mypy
test : venv java-requirement
	source venv/bin/activate && py.test pdart

smalltest : venv java-requirement
	source venv/bin/activate && py.test pdart/fs pdart/pipeline

aq : venv
	source venv/bin/activate && python AQ.py

mtb : venv
	source venv/bin/activate && \
	    python MakeTarball.py /Users/spaceman/pdart/new-bulk-download . 11187

PROJ_IDS=7240 9296 15419

STEPS=copy_primary_files record_changes insert_changes populate_database \
    build_browse build_labels update_archive make_deliverable

pipeline : venv
	# i WFC3 hst_15419 is 101.5MB
	-rm -rf tmp-working-dir
	tar -xf tmp-working-dir.tar
	for project_id in $(PROJ_IDS); do \
	    for step in $(STEPS); do \
		echo $$project_id $$step; \
		source venv/bin/activate && \
		    python Pipeline.py $$project_id $$step; \
            done; \
        done; \
	say pipeline is done
	open tmp-working-dir

tar : tmp-working-dir.tar
	# pass

TAR_STEPS=download_docs check_downloads
tmp-working-dir.tar :
	-rm -rf tmp-working-dir/*
	for project_id in $(PROJ_IDS); do \
	    for step in $(TAR_STEPS); do \
		source venv/bin/activate && \
		    python Pipeline.py $$project_id $$step; \
	    done; \
        done; \
	say okay
	tar -cf tmp-working-dir.tar tmp-working-dir


BIG_PROJ_IDS=5167 5215 5216 5217 5218 5219 5220 5221 5238 5313 5321 5329 5361 5392 5489 5493 5508 5590 5624 5633 5640 5642 5653 5662 5776 5782 5783 5824 5828 5829 5831 5832 5834 5836 5837 5844 6009 6025 6028 6029 6030 6141 6145 6215 6216 6218 6219 6259 6295 6315 6328 6447 6452 6481 6497 6509 6521 6559 6591 6621 6630 6634 6648 6650 6662 6663 6679 6733 6736 6741 6743 6752 6753 6774 6793 6803 6806 6818 6841 6842 6846 6852 6853 7240 7276 7308 7321 7324 7427 7428 7429 7430 7589 7594 7616 7717 7792 7916 8148 8152 8169 8274 8391 8398 8405 8577 8579 8580 8583 8634 8660 8680 8699 8800 8802 8871 8876 9052 9059 9060 9235 9256 9259 9268 9296 9302 9320 9341 9344 9354 9384 9385 9391 9393 9426 9440 9508 9585 9678 9713 9725 9738 9745 9746 9747 9748 9809 9823 9975 9985 9991 10065 10095 10102 10115 10140 10144 10156 10165 10170 10192 10268 10357 10398 10422 10423 10427 10456 10461 10468 10473 10502 10506 10507 10508 10512 10514 10534 10545 10555 10557 10625 10719 10770 10774 10781 10782 10783 10786 10799 10800 10801 10805 10860 10862 10870 10871 10992 11055 11085 11096 11102 11109 11113 11115 11118 11156 11169 11170 11178 11187 11226 11292 11310 11314 11361 11418 11497 11498 11518 11536 11556 11559 11566 11573 11630 11644 11650 11656 11806 11956 11957 11969 11970 11971 11972 11984 11990 11998 12003 12045 12049 12053 12077 12119 12176 12234 12237 12243 12245 12305 12395 12435 12436 12463 12468 12535 12537 12597 12601 12660 12665 12675 12725 12792 12801 12887 12891 12894 12897 12980 13005 13012 13031 13051 13055 13067 13118 13198 13199 13229 13311 13315 13404 13414 13438 13474 13475 13502 13503 13609 13610 13612 13620 13631 13632 13633 13663 13664 13667 13668 13675 13692 13712 13713 13716 13794 13795 13829 13863 13864 13865 13866 13873 13934 13936 13937 14040 14042 14044 14045 14053 14064 14092 14103 14113 14133 14136 14138 14192 14195 14217 14261 14263 14334 14458 14474 14475 14485 14491 14492 14498 14499 14524 14616 14627 14629 14661 14752 14756 14790 14798 14839 14864 14884 14928 14936 14939 15097 15098 15108 15142 15143 15144 15158 15159 15171 15207 15233 15248 15259 15261 15262 15328 15342 15343 15344 15357 15360 15372 15405 15406 15409 15419 15421 15423 15447 15450 15456 15460 15481 15492 15493 15500 15502 15505 15581 15595 15622 15623 15648 15665 15678 15706 15821 15929 16009  

BIG_STEPS=download_docs check_downloads $(STEPS)

big-pipeline : venv
	for project_id in $(BIG_PROJ_IDS); do \
	    for step in $(BIG_STEPS); do \
		echo $$project_id $$step; \
		source venv/bin/activate && \
		    python Pipeline.py $$project_id $$step; \
            done; \
        done;

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
