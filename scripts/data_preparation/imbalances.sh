#!/bin/bash

if [ "$#" -lt 2 ]
then
    echo "Usage:"
    echo "./imbalances.sh ../../data2/clamp/train.txt ../../task/clamp/none/train_output.log"
    exit
fi

for file in $1 $2
do
    if [ ! -f $file ]
    then
	echo "no such file: "$file
	exit
    fi
done

# 2 new lines
sed -i -e '$a\' $2
sed -i -e '$a\' $2

grep 'Red.*1$' $1 | wc -l | xargs -i echo "Redbox positives: {}" >> $2
grep 'Red.*0$' $1 | wc -l | xargs -i echo "Redbox negatives: {}" >> $2
grep 'Blue.*1$' $1 | wc -l | xargs -i echo "Bluebox positives: {}" >> $2
grep 'Blue.*0$' $1 | wc -l | xargs -i echo "Bluebox negatives: {}" >> $2

