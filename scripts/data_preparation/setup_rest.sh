#!/bin/bash
#set -e
# any subsequent command that fails will exit the script


# set variables below in order
# to create caffe/data/$BASE_NAME
#           caffe/data_info/$BASE_NAME
#           caffe/models/$FULL_NAME

# BASE and FULL name in case 2 models share same data & labels

BASE_NAME=$1
NUM_OUTPUT=${2:-2}
FULL_NAME=${3:-$BASE_NAME}

echo "Running setup_rest with BASE_NAME:"$BASE_NAME" FULL_NAME:"$FULL_NAME" and NUM_OUTPUT:"$NUM_OUTPUT

cd ../data_preparation



# 3. create leveldb inputs
cd /data2/ad6813/caffe/models

# first make sure exists reference dir from which to cp and sed
if [ -d clampdetBase ]
then
    rm -rf $BASE_NAME
    mkdir $BASE_NAME
    cd clampdetBase

    NEEDED_FILES="clampdet_solver.prototxt create_clampdet.sh fine_clampdet.sh clampdet_train.prototxt make_clampdet_mean.sh clampdet_val.prototxt resume_training.sh"
    for file in $NEEDED_FILES;
    do
	if [ ! -f $file ]
	then
	    echo "$file not found"
	    echo "need it to create leveldb inputs for $BASE_NAME"
	    exit
	else
	    cp $file '../'$BASE_NAME'/'
	fi
    done
else
    echo "directory clampdet not found"
    echo "need it to create leveldb inputs for $BASE_NAME"
    exit
fi

# now adapt files to taskname
cd ../$BASE_NAME
# rename files
for file in *clampdet*;
do mv $file ${file/clampdet/$BASE_NAME};
done
# modify contents of files
for file in *; do sed -i 's/clampdet/'$BASE_NAME'/g' $file; done
'./create_'$BASE_NAME'.sh'


# 4. compute mean image
echo "computing mean image..."
'./make_'$BASE_NAME'_mean.sh'
if [ ! -f '../../data/'$BASE_NAME'/'$BASE_NAME'_mean.binaryproto' ]
then
    scp graphic06.doc.ic.ac.uk:/data2/ad6813/caffe/data/clampdet/clampdet_mean.binaryproto '../../data/'$BASE_NAME'/'$BASE_NAME'_mean.binaryproto'
fi


# 5. network definition
# keeping batchsize 50
for TYPE in train val;
do
    # change net name and num neurons in output layers
    sed -i $BASE_NAME'_'$TYPE'.prototxt' -e '1s/Clamp/'$BASE_NAME'/' -e '300s/2/'$NUM_OUTPUT'/';
done


# 6. solver
sed -i $BASE_NAME'_solver.prototxt' -e '10s/20000/'$MAX_ITER'/' -e '13s/2000/'$SNAPSHOT'/'


# 7. go!
chmod 755 ./fine_"$BASE_NAME".sh
mkdir logs
echo "you're ready!"
echo "cd ../../models/"$BASE_NAME""
echo "nohup ./fine_"$BASE_NAME".sh >> train_output.txt 2>&1 &"

