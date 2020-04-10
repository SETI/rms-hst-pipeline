#!/bin/bash

function usage(){
    echo 'Usage: grep-placeholders.sh <TARFILE>'
    exit 1
}

# Collect the argument or give an error message.

if [ $# -ne 1 ]; then
    usage
else
   TARFILE=$1
fi

# Create a temporary directory.

TMPDIR=$(mktemp -d -t grep-placeholders)
if [[ ! "$TMPDIR" || ! -d "$TMPDIR" ]]; then
    echo 'grep-placeholders.sh: mktemp failed.'
    exit 1
fi

# Set up the temporary directory to self-delete when exiting, whether
# by error or regularly.

function cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

# Uncompress the tarfile into the temporary directory and move into it.

tar -xf $TARFILE -C $TMPDIR
cd $TMPDIR

# (1) Look through all the labels (XML files) in the archive, looking
# for either the three hash-marks that single placeholders or the
# string "magrathea" (that indicates a failed target identification.
# (2) Use awk to trim leading and trailing space and collapse internal
# space.  (3) Sort the lines and remove duplicates.

grep -h -E -R --include="*.xml" '(magrathea|\#\#\#)' . | \
    awk '{$1=$1;print}' | \
    sort -u



