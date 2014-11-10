#!/bin/bash
set -e

TRAIN_FN=$1
NUM_FULL_COPIES=$2
LAST_COPY=$3

echo "Running over_sample.sh with TRAIN_FN:"$1" NUM_FULL_COPIES:"$2" and LAST_COPY:"$3

echo "First there were "$(grep '1$' $1 | wc -l)" positives in train.txt"

# get all minority class cases
grep '1$' $1 > full_copy
sort -R full_copy > tempf
mv tempf full_copy

# append a partial copy of them to train file
cat full_copy | head -$LAST_COPY > last_copy

# append it all to train file
./append.py full_copy $1 $2
./append.py last_copy $1 1

# shuffle train file
echo "shuffling train file..."
sort -R $1 > tempf
mv tempf $1

# rm {full_copy, last_copy}


