#!/bin/bash

cd $1"/raw_data/dump"
grep "JointId" *.met | tr ":=." " " | awk '{a[$4] = a[$4]" "$1}; END{for (val in a) print val"\t"a[val]}' > "../../multJoints"
