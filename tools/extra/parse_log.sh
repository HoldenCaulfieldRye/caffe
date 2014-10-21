#!/bin/bash
# Usage parse_log.sh caffe.log
# It creates the following two text files, each containing a table:
#     caffe.log.test (columns: '#Iters Seconds TestAccuracy TestLoss')
#     caffe.log.train (columns: '#Iters Seconds TrainingLoss LearningRate')


# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your.log"
exit
fi
LOG=`basename $1`
grep -B 1 'Test ' $1 > aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux_iter.txt
grep 'Test loss' aux.txt | awk '{print $7}' > aux_tloss.txt
grep 'Test net output #0' aux.txt | awk '{print $11}' > aux_acc0.txt
grep 'Test net output #1' aux.txt | awk '{print $11}' > aux_acc1.txt
grep 'Test net output #2' aux.txt | awk '{print $11}' > aux_acc2.txt
grep 'Test net output #3' aux.txt | awk '{print $11}' > aux_acc3.txt

# Extracting elapsed seconds
# For extraction of time since this line contains the start time
grep '] Solving ' $1 > aux1.txt  # aux3
grep 'Testing net' $1 >> aux1.txt
$DIR/extract_seconds.py aux1.txt aux_sec.txt # aux3.txt aux4.txt

# Generating
# echo '#Iters Seconds TestAccuracy TestLoss'> $LOG.test
echo '#Iters Seconds      TestLoss   Acc_0         Acc_1      PCAcc      Accuracy '> $LOG.test
paste aux_iter.txt aux_sec.txt aux_tloss.txt aux_acc0.txt aux_acc1.txt aux_acc2.txt aux_acc3.txt | column -t >> $LOG.test
rm aux.txt aux_iter.txt aux_sec.txt aux_tloss.txt aux_acc0.txt aux_acc1.txt aux_acc2.txt aux_acc3.txt

# For extraction of time since this line contains the start time
grep '] Solving ' $1 > aux.txt
grep ', loss = ' $1 >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ', loss = ' $1 | awk '{print $9}' > aux1.txt
grep ', lr = ' $1 | awk '{print $9}' > aux2.txt

# Extracting elapsed seconds
$DIR/extract_seconds.py aux.txt aux3.txt

# Generating
echo '#Iters Seconds TrainingLoss LearningRate'> $LOG.train
paste aux0.txt aux3.txt aux1.txt aux2.txt | column -t >> $LOG.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt


