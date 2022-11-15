#
# The HST Migration Pipeline Architecture

Version 2.0

November 9, 2022

#
# Terms

- Program ID = the four or five-digit numeric HST proposal ID.
- PPPSSOOT = the format of the first nine characters in the root name of a file. I indicates instrument, PPP indicates program, SS for visit, OO for observation, T for transmission/association.
- IPPPSSOO = the first eight characters. On occasion, files have the same “IPPPSSOO” but the “T” character is different. Nevertheless, files with the same IPPPSSOO are always associated, regardless of the “T”.
- VISIT = the two character designation for the HST visit, the “SS” in IPPPSSOOT.
- TRL: File suffix that describes the processing history of all the files with the same IPPPSSOOT prefix.
- SPT/SHM/SHF: File suffix that identifies the target for each IPPPSSOOT prefix. These are often collectively referred to as “SPT” files, even if the suffixes differ.

#
# "Logically Complete"

- The “logically complete” set of data files is one in which every file to which another file might refer is also in the set.
  - The pipeline will produce incorrect results if an attempt is made to run it on sets of files that are not logically complete.
- Often, all the files with a single IPPPSSOO are logically complete.
- For files with associations (e.g., repeat and CR-split exposures), the logically complete set is defined by the union of:
  - the IPPPSSOO of the association file “\*\_asn.fits”
  - every other file with that IPPPSSOO.
  - every file with an IPPPSSOO referenced within the association file.
- You can recognize these cases because the “T” in IPPPSSOOT is a digit, usually 0. Under other circumstances, the “T” is never a digit.
- Because associated files never cross a visit boundary, it probably makes most sense to focus on archiving files by visit.
- A visit is limited to ~ 6 orbits of HST, so this approach has the advantage of also limiting the volumes of data that the pipeline has to deal with at any given time.

#
# File System Organization (1)

- I suggest we organize all information about the processing pipeline by program ID and visit within a single directory tree.
- The directory path is always `<root>/hst_\nnnnn>/visit_<ss>/`, where nnnnn is the program ID (with leading zero if needed) and ss is the visit.
- The root directory is always defined by the content of an environment variable “`HST_PIPELINE`”.
- All processing logs are separated by visit and saved in within a subdirectory “`logs/`” inside each visit-level directory.
- Any other information that needs to be managed by the pipeline can be saved in files within this tree.

#
# File System Organization (2)

- We should not use this directory tree to store data files and their labels while in development.
- I suggest we use the same hierarchy, but have them in a root directory defined by environment variable “`HST_STAGING`”.
- The pipeline also needs to reference the location of all current HST bundles as defined by environment variable “`HST_BUNDLES`”.
- See the document PDS4-VERSIONING.txt on Dropbox for the details about the layout of the `HST_BUNDLES` tree.
- Note: It probably makes sense to move the target identification WEBCACHE out of GitHub and into this directory tree as `<HST_PIPELINE>/target-identifications/WEBCACHE`.

#
# What is a Task?

- A task is a single stand-along program that executes some piece of the pipeline and then stops. It can be typed as a command at the terminal if desired.
- Every time a task runs, it creates a unique log file describing what it has done.
- Tasks are designed to be relatively independent of one another, so that multiple tasks can run in parallel, using multiple threads, without interfering with one another.
- Log files are given the name of the task, followed by “`-yyyy-mm-ddThh-mm-ss.log`” containing the local time at which the task started in the format year-month-day-hour-minute-second. They are saved in “`logs/`” subdirectories at various levels in the file system hierarchy.
- The pipeline is “queue-based”, meaning that it maintains a queue of available tasks.
- Just before completion, tasks typically generate one or more new tasks and add them to the task queue.

#
# The Task Queue

- The design and implementation of the Task Queue and the Queue Manager is TBD.
- Requirements include:
  - A task can add one or more new tasks to the queue, with a specified delay or level of priority, and then proceed asynchronously.
  - A task can add one or more new tasks to the queue, with a specified level of priority, and then wait for them to finish.
  - A task can ask the queue manager if a task that it sent to the queue manager is now complete.
- Priorities are 1 (Lowest) to 5 (Highest).
  - Priorities are assigned so that once we begin processing an HST bundle, we prioritize finishing it over starting a different bundle.
- Rob has agreed to research Queue Manager options for us.

#
# Overview of Tasks

- **query-hst-moving-targets** \<query constraints\>
  - runs periodically (or can be run manually) to get a list of programs with moving targets.
- **query-hst-products** \<program-ID\>
  - get a complete list of the accepted files for a specified program.
- **update-hst-program** \<program-ID\> \<visit\> [\<visit\> …]
  - overall task to create a new bundle or to manage the update of an existing bundle.
- **get-program-info** \<program-ID\>
  - retrieve the online files that describe a program (such as .apt or .pro) and assemble other program-level information.
- **retrieve-hst-visit** \<program-ID\> \<visit\>
  - retrieve the accepted FITS files and browse products from MAST.
- **label-hst-products** \<program-ID\> \<visit\>
  - create labels for the FITS data files.
- **prepare-browse-products** \<program-ID\> \<visit\>
  - create and label the browse products.
- **finalize-hst-bundle** \<program-ID\>
  - package a complete set of files in the staging directories as a new bundle or as updates to an existing bundle

#
# Work Flow

![flow chart](workflow.png)

#
# Task: **query-hst-moving-targets**

- Initiated by: **query-hst-moving-targets**, or manually.
- Input:
  - Optional range of instruments, dates, or program IDs.
  - Note: Command line syntax can be largely based on what is in Dave’s current version of query-mast.py.
- Log: `<HST_PIPELINE>/logs/query-hst-moving-targets-<ymdhms>`.logs.
- Priority: 1 (Lowest)
- State:
  - There is always one `<HST*PIPELINE>/hst*<nnnnn>` directory for each HST program that is known to contain planetary observations.
- Actions:
  - Query MAST for any program IDs that meet the query constraints.
  - For each subdirectory of `<HST_PIPELINE>` that is missing, queue task **query-hst-products** for that program ID.
  - Also re-queue task **query-hst-moving-targets** with a 30-day delay.

#
# Task: **query-hst-products** (1)

- Initiated by: **query-hst-moving-targets**, **query-hst-products**, or manually.
- Input: program ID
- Log: `<HST_PIPELINE>/hst_<nnnnn>/logs/query-hst-products-<ymdhms>.log`
- Priority: 1 (Lowest)
- State:
  - File `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/products.txt` contains the list of available products with accepted suffixes, sorted alphabetically, for each visit. Missing on first run.
  - File `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/trl_checksums.txt` contains the current list of all TRL files and their checksums. Missing on first run.
- Actions:

  - Query MAST for all available visits and files in this program.
  - Create `<HST_PIPELINE>/hst_<nnnnn>/` and any `visit_<ss>/` subdirectories that do not already exist.
  - Download all the TRL files for this HST program to `<HST_STAGING>/hst_<nnnnn>/`.

#
# Task: **query-hst-products** (2)

- Actions (continued):
  - For each visit…
    - If `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/products.txt` already exists, check its contents against the list from MAST and identify any changes.
    - If changes are identified, move products.txt to `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/backups/products-<ymdhms>.txt` and replace the content of products.txt.
    - If `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/trl_checksums.txt` does not already exist, create it with the checksums of the staged TRL files.
    - Otherwise, compare the checksums. If there are any changes, rename `trl_checksums.txt` to `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/backups/trl_checksums-<ymdhms>.txt` and save the new content.
    - Delete the TRL files.
  - Create a list of visits in which any files are new or changed.
  - If the list is not empty, queue task **update-hst-program** with the list of visits.
  - If the list was empty, re-queue task **query-hst-products** with a 30-day delay.
  - Otherwise, re-queue task **query-hst-products** with a 90-day delay.

#
# Task: **update-hst-program**

- Initiated by: **query-hst-products**
- Input: program ID, list of visits.
- Log: `<HST_PIPELINE>/hst_<nnnnn>/logs/update-hst-program-<ymdhms>.log`
- Priority: 2 (Low)
- State:
  - File `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/products.txt` is up to date for each visit.
- Actions:
  - Queue task **get-program-info** and wait for it to complete.
  - For each visit, queue task **update-hst-visit** and wait until all the visits have completed.
  - Queue task: **finalize-hst-bundle** and wait for it to finish.
  - Send some sort of notification.

#
# Task: **get-program-info**

- Initiated by: **update-hst-program**
- Input: program ID
- Log: `<HST_PIPELINE>/hst_<nnnnn>/logs/get-program-info-<ymdhms>.log`
- Priority: 5 (Highest)
- State:
  - Directory `<HST_PIPELINE>/hst_<nnnnn>/` contains the proposal files `<nnnnn>.apt`, `.pro`, and/or `.prop` files, plus the `.pdf` version. Absent on first run.
  - File `<HST_PIPELINE>/hst_<nnnnn>/program-info.txt` contains citation info and maybe other info about the program overall. Absent on first run.
- Actions:
  - Retrieve the proposal files via a web query.
  - If these files are the same as the existing ones, stop.
  - Otherwise,
    - Rename each existing file by appending “`-<ymdhms>`” before its extension and moving it to a `backups/` subdirectory.
    - Then save the newly downloaded files.
    - Regenerate `program-info.txt` with a possibly modified citation.

#
# Task: **update-hst-visit**

- Initiated by: **update-hst-program**
- Input: program ID, visit.
- Log: `<HST_PIPELINE>/hst_<nnnnn>/logs/update-hst-visit<ymdhms>.log`
- Priority: 3 (Medium)
- Actions:
  - Queue **retrieve-hst-visit** for this visit and wait for it to complete.
  - Queue **label-hst-products** for this visit and wait for it to complete.
  - Queue task **prepare-browse-products** for this visit and wait for it to complete.

#
# Task: **retrieve-hst-visit**

- Initiated by: **update-hst-visit**
- Input: program ID and visit
- Log: `<HST*PIPELINE>/hst*<nnnnn>/visit\_<ss>/logs/retrieve-hst-visit-<ymdhms>.log`
- Priority: 4 (High)
- State:
  - The file `<HST*PIPELINE>/hst*<nnnnn>/visit\_<ss>/products.txt` always contains the list of available products (FITS files or browse products) with approved suffixes.
- Actions:
  - Retrieve all the identified files and put them in `<HST*STAGING>/hst*<nnnnn>/visit\_<ss>/`.

#
# Task: **label-hst-products**

- Initiated by: **update-hst-visit**
- Input: program ID and visit
- Log: `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/logs/label-hst-products-<ymdhms>.log`
- Priority: 5 (Highest)
- State:
  - New files are in `<HST_STAGING>/hst_<nnnnn>/visit_<ss>/`.
  - Existing bundle, if any, is at `<HST_BUNDLES>/hst_<nnnnn>/`.
- Actions:
  - Compare the staged FITS files to those in an existing bundle, if any.
  - Create a new XML label for each file.
  - Reset the modification dates of the FITS files to match their production date at MAST.
  - If any file contains NaNs, rename the original file with “`-original`” appended, and then rewrite the file without NaNs.
  - TBD: What to do if the file in the existing bundle is identical.

#
# Task: **prepare-browse-products**

- Initiated by: **update-hst-visit**
- Input: program ID and visit
- Log: `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/logs/prepare-browse-products-<ymdhms>.log`
- Priority: 5 (Highest)
- State:
  - Files and labels are in `<HST_STAGING>/hst_<nnnnn>/visit_<ss>/`.
- Actions:
  - Prepare the browse products and their labels, and save them in Files and labels are in `<HST_STAGING>/hst_<nnnnn>/visit_<ss>/`.
  - Details TBD.

#

# Task: **finalize-hst-bundle**

- Initiated by: **update-hst-program**
- Input: program ID
- Log: `<HST_PIPELINE>/hst_<nnnnn>/visit_<ss>/logs/finalize-hst-bundle-<ymdhms>.log`
- Priority: 5 (Highest)
- State:
  - Files and labels are in `<HST_STAGING>/hst_<nnnnn>/visit_<ss>/`.
  - All the files and labels inside `<HST_STAGING>/hst_<nnnnn>` and its visit subdirectories are up to date.
- Actions:
  - Create the documents, schema, context, kernel? directories.
  - Move existing, superseded files as described in PDS4-VERSIONING.txt
  - Move the new files into their proper places in `<HST_BUNDLES>/hst_<nnnnn>/`.
  - Create the new `collection.csv` and `bundle.xml` files.
  - Run the validator.
  - Question: Will we need this process to generate a doi? How do we handle that?
