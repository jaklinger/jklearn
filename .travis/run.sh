#!/bin/bash

set -e
set -x

# Setup test data
TOPDIR=$PWD

cd $TOPDIR

# Run every test
for TOPDIRNAME in clusters;
do
    TESTDIRS=$(find jklearn/$TOPDIRNAME -name "test*" -type d)
    for TESTDIRNAME in $TESTDIRS;
    do
	python -m unittest discover $TESTDIRNAME
    done
done
