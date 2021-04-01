# type: ignore
#!/usr/bin/python
"""
Created on November 6, 2018

Input: directory of PDS4 products with Product_Bundle in root
Output (lid_reference below includes lidvid_reference):
- List every lid_reference with no corresponding logical_identifier in input
- List every lid_reference with count of referrers (verbose)
- List of local_identifier_reference with no local_identifier  LATER
- If Product_Bundle given, check all labels in that tree have
  urn:nasa:pds:<bundle_id>:...
- If Product_Collection given, check all labels in that tree have
  urn:nasa:pds:<bundle_id>:<collection_id>:<product_id>
@author: rchen
"""

import string, sys, re, getopt
import os  # rename file
import json
import re
import datetime
import StringIO

verbose = False
ignoreOld = False
LOG_DIR = "lidvid_error_log.txt"


def usage():
    print("usage: ", sys.argv[0], "[-hv] <contextProductDirectory> [<...>]")
    print("  -h this help")
    print("  -i ignore references from old versions of files")
    print("  -v verbose")
    print("  -x dir with existing context files. If none, outputs have version_id==1.0")
    print("validate complete set of context products")
    sys.exit()


try:
    opts, args = getopt.getopt(sys.argv[1:], "hiv", ["help", "ignore", "verbose"])
except getopt.GetoptError as err:
    print(str(err))  # will print something like "option -a not recognized"
    usage()
for o, a in opts:
    if o in ("-h", "--help"):
        usage()
    elif o in ("-i", "--ignore"):
        ignoreOld = True
    elif o in ("-v", "--verbose"):
        verbose = True
    else:
        assert False, "unhandled option"
if len(args) < 1:
    usage()


def flattenFilesOrDirs(paths):
    toReturn = []
    for p in paths:
        if (
            re.match("\.", p)  # allow initial ./xxx
            and p is not "."
            and not re.match("\.\/", p)
            and p is not ".."
            and not re.match("\.\.\/", p)
        ):
            continue
        if os.path.isdir(p):
            for inFile in os.listdir(p):
                if re.match("\.", inFile):
                    continue  # skip .DS_Store
                x = flattenFilesOrDirs([p + "/" + inFile])
                if x:
                    toReturn.extend(x)
        else:
            toReturn.append(p)
    return toReturn


lid2viddef = {}  # lid2viddef[LID] = dict of vid:fileDefiningLID
lid2refs = {}  # lid2refs[LID] = list of files referring to LID
lidvid2refs = {}  # lidvid2refs[LID::VID] = list of files referring to LIDVID


def parseFiles(allFiles):
    # ET handles namespaces verbosely, so this never matches:
    #  tagLid = "Identification_Area/logical_identifier"
    tagLid = "{http://pds.nasa.gov/pds4/pds/v1}Identification_Area/{http://pds.nasa.gov/pds4/pds/v1}logical_identifier"
    tagVid = "{http://pds.nasa.gov/pds4/pds/v1}Identification_Area/{http://pds.nasa.gov/pds4/pds/v1}version_id"
    tagLidRef = "{http://pds.nasa.gov/pds4/pds/v1}lid_reference"
    tagLidvidRef = "{http://pds.nasa.gov/pds4/pds/v1}lidvid_reference"
    lidTree = {}  # temp storage; if ignoreOld and vid is old, dump this
    lidvidTree = {}  # temp storage; if ignoreOld and vid is old, dump this
    lidvid2file = {}  # temp storage
    import xml.etree.ElementTree as ET

    for aFile in allFiles:  # build lidRefs{} and lids{}
        if not re.search("\.xml$", aFile):
            continue
        tree = ET.parse(aFile)
        doc = tree.getroot()
        if doc.find(tagLid) is None:
            print("INFO no LID in", aFile)
            continue
        lid = doc.find(tagLid).text
        vid = doc.find(tagVid).text
        if lid in lid2viddef:
            if vid in lid2viddef[lid]:
                sys.stderr.write(
                    "ERROR duplicate LIDVID "
                    + lid
                    + "::"
                    + vid
                    + "\n  "
                    + aFile
                    + "\n  "
                    + lid2viddef[lid][vid]
                    + "\n"
                )
                continue
        else:
            lid2viddef[lid] = {}
        lid2viddef[lid][vid] = {}
        lid2viddef[lid][vid] = aFile
        # this populates lid2viddef{}{}
        lidTree[lid + "::" + vid] = doc.findall(".//" + tagLidRef)
        lidvidTree[lid + "::" + vid] = doc.findall(".//" + tagLidvidRef)
        lidvid2file[lid + "::" + vid] = aFile
        # now populate the other two dicts
    for lid in sorted(lid2viddef.iterkeys()):
        for vid in sorted(lid2viddef[lid].iterkeys(), reverse=True):
            lv = lid + "::" + vid
            for lidRef in lidTree[lv]:
                if lidRef.text not in lid2refs:
                    lid2refs[lidRef.text] = []
                lid2refs[lidRef.text].append(lidvid2file[lv])  # set lid2refs{}[]
            for lidvidRef in lidvidTree[lv]:
                if lidvidRef.text not in lidvid2refs:
                    lidvid2refs[lidvidRef.text] = []
                lidvid2refs[lidvidRef.text].append(lidvid2file[lv])  # lidvid2refs{}[]
            if ignoreOld:
                break  # quit the vid loop after biggest vid
    return  # how to pass lid2viddef{}, lid2refs{}, lidvid2refs{}?


if __name__ == "__main__":
    allFiles = flattenFilesOrDirs(args)
    parseFiles(allFiles)  # sets lid2viddef{}, lid2refs{}, lidvid2refs{}
    # Record all warning messages
    log_arr = []
    separater = "==========" + args[1] + "=========="
    log_arr.append(separater)
    # Include all lidvid in tmp-context-products.json
    if os.path.isfile("tmp-context-products.json"):
        with open("tmp-context-products.json", "r+") as req_json:
            data = json.load(req_json)
            for context_product in data["Product_Context"]:
                lidvid = context_product["lidvid"]
                lidvid2refs[lidvid] = {}
                idx = lidvid.find("::")
                if idx != -1:
                    lid = lidvid[:idx]
                    vid = lidvid[-3:]
                    lid2viddef[lid] = {}
                    lid2viddef[lid][vid] = {}
    # Include handbook lid:
    data_handbook_lid = "urn:nasa:pds:hst-support:document:acs-dhb"
    inst_handbook_lid = "urn:nasa:pds:hst-support:document:acs-ihb"
    for lid in [data_handbook_lid, inst_handbook_lid]:
        vid = "::1.0"
        livid = lid + vid
        lidvid2refs[lidvid] = {}
        lid2viddef[lid] = {}
        lid2viddef[lid][vid] ={}

    if verbose:
        print("___CHECK IF EVERY lid_reference IS DEFINED___")
    for lid in sorted(lid2refs.iterkeys()):
        if lid not in lid2viddef:
            msg = (
                "WARNING "
                + str(len(lid2refs[lid]))
                + " lid_references to absent LID "
                + str(lid)
            )
            print(msg)
            log_arr.append(msg)
            if verbose:
                for rFile in lid2refs[lid]:
                    print(" ", rFile)
    if verbose:
        print("___CHECK IF EVERY lidvid_reference IS DEFINED___")
    for lidvid in sorted(lidvid2refs.iterkeys()):
        lid, vid = lidvid.split("::")
        if lid in lid2viddef:
            if vid in lid2viddef[lid]:
                continue  # all good
            else:
                msg = (
                    "WARNING "
                    + str(len(lidvid2refs[lidvid]))
                    + " lidvid_references to absent VID "
                    + str(lidvid)
                )
                print(msg)
                log_arr.append(msg)
        else:
            msg = (
                "WARNING "
                + str(len(lidvid2refs[lidvid]))
                + " lidvid_references to absent LID "
                + str(lidvid)
            )
            print(msg)
            log_arr.append(msg)
        if verbose:
            for rFile in lidvid2refs[lidvid]:
                print("   ", rFile)
    # Should probably merge the 2 DUMPs by chopping the VID out of lidvid2refs,
    if verbose:
        print("___DUMP EVERY lid_reference + COUNT OF REFERRERS___")
        for lid in sorted(lid2refs.iterkeys()):
            print(lid, "\t", len(lid2refs[lid]))
            # list all referers? That's too much
        print("___DUMP EVERY lidvid_reference + COUNT OF REFERRERS___")
        for lidvid in sorted(lidvid2refs.iterkeys()):
            print(lidvid, "\t", len(lidvid2refs[lidvid]))
    if verbose:
        print("___CHECK IF EVERY local_identifier_reference IS DEFINED___")
    if verbose:
        print("___CHECK THAT ALL LIDS ARE HIERARCHICAL___")
    # alphabetically, bundleLID is first, and a collectionLID precedes its prods
    bundleLID = None
    collectionLID = None
    for lid in sorted(lid2viddef.iterkeys()):
        m = re.match("(urn:[^:]*:[^:]*:[^:]*)(:([^:]*)(:[^:]*)?)?$", lid)
        if bundleLID and m.group(1) == bundleLID:
            if collectionLID and m.group(3) == collectionLID:
                continue  # normal
            elif m.group(4):
                msg = "WARNING deriving collection LID from " + str(lid)
                print(msg)
            # else setting collectionLID after bundle product
            collectionLID = m.group(3)
            if verbose:
                print("INFO collection: ...:" + collectionLID)
        else:
            if m.group(2):
                msg = "WARNING deriving bundle LID from " + str(lid)
                print(msg)
            # else setting initial bundleLID, which is fine
            bundleLID = m.group(1)
            if verbose:
                print("INFO bundle:", bundleLID)
            if m.group(3):
                collectionLID = m.group(3)
                if verbose:
                    print("INFO collection: ...:" + collectionLID)
            else:
                collectionLID = None  # happens iff bundle product

    # Write all *to absent* warnings into lidvid_error_log.txt
    with open(LOG_DIR, "a") as f:
        for log in log_arr:
            f.write("%s\n" % log)
    sys.exit()
