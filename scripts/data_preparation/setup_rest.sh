#!/bin/bash
#set -e
# any subsequent command that fails will exit the script


##################################################################
### commented code which may be useful one day
##################################################################


# NOTE!
# selecting which labels to learn, whether to merge or rename is not
# specified at command line, so this script is not fully automatic.
# however, these prompts will occur at the beginning only, so as long
# as you have a single task ie a single string in BASE_NAME, this
# script is still useful, all you have to do is reply to prompts.

# SIZE="expr $(cat ../../data_info/$BASE_NAME/train.txt | wc -l) + $(cat ../../data_info/$BASE_NAME/val.txt | wc -l) + $(cat ../../data_info/$BASE_NAME/test.txt | wc -l)"
# echo $($SIZE)

# with 4, bad minimum provides 80% classification accuracy
# read -p "Target bad min? (e.g. 0.8 for class imbalance such that 80% a bad/fake minimum yields 80% accuracy) "
# read -p "Max num minibatch passes for training? (20000, cos 10500 was optimal for clampdet) "
# delete this one once you have cuda-convnet style snapshotting
# read -p "Network snapshot frequency? (2000) "

# variables for setup.py
# USE_FIRST_LOCAL_DICT=Y

# # 1. & 2. get labels, choose which ones to learn, symlink dataset
# source /data/ad6813/caffe/python/venv/bin/activate
# if [ ! -d /data/ad6813/caffe/data_info/$BASE_NAME ]
# then
#     mkdir /data/ad6813/caffe/data_info/$BASE_NAME
# fi
# if [ ! -d /data/ad6813/caffe/data/$BASE_NAME ]
# then
#     mkdir /data/ad6813/caffe/data/$BASE_NAME
# fi
# echo "main and move_to_dirs..."
# # NUM_OUTPUT is number of classes to learn
# NUM_OUTPUT=$(python setup_data.py data-dir=/data/ad6813/pipe-data/Bluebox/raw_data/dump data-info=/data/ad6813/caffe/data_info/$TASK_NAME to-dir=/data/ad6813/caffe/data/$TASK_NAME bad-min=N)
# echo "number of output neurons: "$NUM_OUPUT



##################################################################
### BEGIN
##################################################################

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

# 1. resize images
cd /data/ad6813/caffe/data/$BASE_NAME
CMD=$(convert train/$(ls train | tail -1) -print "%wx%h" /dev/null) 
if [ $CMD != "256x256" ]
then
    echo "downsizing all images to 256x256..."
    for name in */*.jpg; do convert -resize 256x256\! $name $name; done
else
    echo "images already downsized"
fi


# 2. download alexnet
if [ -f /data/ad6813/caffe/models/alexnet/caffe_alexnet_model ]
then echo "alexnet already downloaded"
else
    cd /data/ad6813/caffe/models/alexnet
    ./get_caffe_alexnet_model.sh
    echo "repeat to check correct download:"
    ./get_caffe_alexnet_model.sh
    echo "attention: look just above to check correct download"
fi


# 3. create leveldb inputs
cd /data/ad6813/caffe/models

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
    scp graphic06.doc.ic.ac.uk:/data/ad6813/caffe/data/clampdet/clampdet_mean.binaryproto '../../data/'$BASE_NAME'/'$BASE_NAME'_mean.binaryproto'
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

