
TROUBLESHOOT
============

cmd:
./build/tools/caffe train -solver models/clampdet/solver.prototxt \
-weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

err:
I1013 18:51:06.336516 26677 net.cpp:67] Creating Layer data
I1013 18:51:06.336534 26677 net.cpp:356] data -> data
I1013 18:51:06.336555 26677 net.cpp:356] data -> label
I1013 18:51:06.336573 26677 net.cpp:96] Setting up data
I1013 18:51:06.336585 26677 image_data_layer.cpp:30] Opening file data/clampdet/train.txt
I1013 18:51:06.336643 26677 image_data_layer.cpp:45] A total of 0 images.
Segmentation fault

sol:
data/<name>/{train,val}.txt missing label numbers


dirty protobuf hack:
layer_factory.cpp: case OLD: return IMPOSTOR
loss_layers.hpp:   IMPOSTOR::LayerParameter_LayerType() { return OLD }


cleaner protobuf hack:
caffe.proto: give new layer smaller ID, swap



TROUBLESHOOT on CAFFE OLD
=========================

# during make all:
/usr/bin/ld: cannot find -lcblas
/usr/bin/ld: cannot find -latlas
# solution:
scp graphic06.doc.ic.ac.uk:/etc/alternatives/lib*las* ~/.local/lib

# create image mean:
Check failed: proto.SerializeToOStream(&output)
# hack solution:
use a sufficiently similar, previously computed image mean
# solution:
paths specified in make_*_image_mean.sh do not exist, fix

# threshold layer:
Check failed: (*top)[0]->num() == (*top)[1]->num() (0 vs. 50) The data and label should have the same number.
# solution:
you'd scp -r 'ed the data from another graphic machine, symlinks were
followed, and actual images were in the data dir. that's not really
supposed to be a pb though.


# plot.py: list index out of range
look at log.{train,test} and see if last line pathogenic


# python wrappers:
ImportError: No module named _caffe
# solution
make pycaffe


# leveldb locked:
IO error: lock *_leveldb/LOCK: already held by process
# solution 1
rm -rf *leveldb
./create
# solution 2
{train,val}.prototxt data_param { source: reference correct? }


# inf nan consec Test Score in log:
Test score #32: 0.146749
Test score #33: -0.214647
Test score #34: 0.0004478
Test score #35: -0.312895
[...]
Iteration 1, lr = 9.995e-05
Iteration 1, loss = nan
# solution:
# you have a layer L linking to layer L+1 and L+2 or smth like that


# adding new layer
# need to modify this crazy
"\0011\"\351\001\n\024ConvolutionParameter\"
stuff in build/src/caffe/proto/caffe.pb.cc ?


# ImportError: No module named _caffe
# _<module> usually stands for <module>.so written in C(++)!
cd /data/add6813/caffe
make pycaffe

# pycaffe::_Net_set_mean()
# ValueError: axes don't match array
when it works:
shape of data blob (10, 3, 227, 227)
shape of mean file:  (3, 256, 256)
but for some reason we want mean to have shape:  (3, 227, 227)
when it doesn't:
shape of data blob (10, 3, 227, 227)
shape of mean file:  (1, 3, 256, 256)
but for some reason we want mean to have shape:  (3, 227, 227)
# solution:
mean_f = mean_f[0]


# badmin all over the place on clampdet at some point
ok, just realised:
- clampdet/conv1 bad min (conv2,3 also)
- clampdet/none_reinit no bad min

what the hell? I thought without US, impossible to get no bad min
so what is the magic trick?
-> H1 enable backprop on conv1?
-> H2 re-initialise fc6? 
-> H3 STUPID MISTAKE?
   -> conv1 has train fc7_new, val fc7
      
clampdet/none trains on graphic09
-> no reinit, so if it works, H2 wrong
-> but batchsize 96 might create bad min
-> compare with clampdet/none_reinit as well, just interesting
-> compare with clampdet/tl_wout for Transfer Learning test run
      
looks like it was stupid mistake.
-> so can revert to studying all of transfer learning without under
   sampling
-> so stupid mistake made val error nonsensical, and yet we were
   getting consistent 0.5 pca.
   -> this accuracy layer is still confusing
      need to understand what's going on (?)

clampdet_os/none_reinit:  badmin
-> very confusing. badmin with osampling, not with normal??
-> once again, maybe not badmin, just mistake in oversampling

clampdet/conv1 trained again
-> if works well now, shows stupid mistake last time
-> if so, need to pick more challenging bad min for task 3
   -> show that transfer learning helps tackle imbalance

##



# DEVELOPMENT

attrib                      |  varname       |  meaning
---------------------------------------------------------
prob_.num()                 |  num           |  batchSize
prob_.count()               |                |
prob_.cpu_data()            |  prob_data     |

bottom[1]                   |  
bottom[1]->count()          |

labels_                     |
labels_.count()             |

bottom_diff[case*dimensionality+neuron]
---------------------------------------------------------

the main functions from which net is trained:
":Solve("  	       in src/caffe/solver.cpp
":Forward("            in src/caffe/net.cpp
":Backward("           in src/caffe/net.cpp
":Backward(const"      in src/caffe/layer.hpp
":ComputeUpdateValue(" in src/caffe/solver.cpp
":Update(" 	       in src/caffe/
":Update("             in src/caffe/blob.cpp     (crux)
"void caffe_cpu_axpby(" in src/caffe/util/math_functions.cpp


# conv1
params_[0] dimensions:
num: 96
channels: 3
height: 11
width: 11
count: 34848

params_[1] dimensions:
num: 1
channels: 1
height: 1
width: 96
count: 96

# conv2?
params_[2] dimensions:
num: 256
channels: 48
height: 5
width: 5
count: 307200

params_[3] dimensions:
num: 1
channels: 1
height: 1
width: 256
count: 256

# conv3?
params_[4] dimensions:
num: 384
channels: 256
height: 3
width: 3
count: 884736

params_[5] dimensions:
num: 1
channels: 1
height: 1
width: 384
count: 384

# conv4?
params_[6] dimensions:
num: 384
channels: 192
height: 3
width: 3
count: 663552

params_[7] dimensions:
num: 1
channels: 1
height: 1
width: 384
count: 384

# conv5?
params_[8] dimensions:
num: 256
channels: 192
height: 3
width: 3
count: 442368

params_[9] dimensions:
num: 1
channels: 1
height: 1
width: 256
count: 256

# fc6?
params_[10] dimensions:
num: 1
channels: 1
height: 4096
width: 9216
count: 37748736

params_[11] dimensions:
num: 1
channels: 1
height: 1
width: 4096
count: 4096

# fc7?
params_[12] dimensions:
num: 1
channels: 1
height: 4096
width: 4096
count: 16777216

params_[13] dimensions:
num: 1
channels: 1
height: 1
width: 4096
count: 4096

# fc8?
# softmax weights!
params_[14] dimensions:
num: 1
channels: 1
height: 2   # one for each softmax neuron
width: 4096 # one for each neuron below
count: 8192

params_[15] dimensions:
num: 1
channels: 1
height: 1
width: 2
count: 2



STEP 1
======

solved. you were wrong, fwd pass not ok. move to step 2

debug SBL:
- fwd pass: OK
- bwd pass: OK
- update: PROB
  -> solver.cpp l.250:
     caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->cpu_diff(), momentum,
          history_[param_id]->mutable_cpu_data());
	  
     -> cpu_diff() might be where PROB is
     	-> only for param_id = {14,15} do we have nonzero diff, why??
	   -> because backprop accidentally active on fc8 only
	-> woah! exploding/vanishing cpu_diff() with SBL
	   -> which stage outscales the cpu_diff()? none!
	      what happens to cpu_diff() b4/after bwd pass?
	      add couts in net.cpp l.269
	      -> net.hpp::ForwardBackward calls
	          net::Backward calls
		   layer::Backward calls
		    specific_layer::Backward_cpu
		 solver::ComputeUpdateValue
    		 solver::net_->Update()
	   -> compared w/ benchmark throughout an iteration, similar
	      values (also outscale for benchmark)
	      
   -> actual parameter values might be where PROB is
      -> solver::net_->Update() calls
	     net::Update calls
	       blob::Update
      -> compare logs
	   -> PROB1: layer[10] is 0 for sbl only
	      -> should be ..?
	   -> loss just after ForwardBackward:
	   already fucked up
	   -> cpu_diff just after ForwardBackward: 
		     sbl max       benchmark max
	   net_15    e+34          e+31
	   net_14    0.92          0.99
	   net_13    0.015         0.009
	   net_12    0    !        0.013 (but no neg values!)
	   net_11    0.0055        0.0027
	   -> cpu_diff just after ComputeUpdateValue():
	   fine
	   -> current params, diff, new params:
	   fine
	   -> cpu_diff just after Update():
	   fine
	   
  -> so the fucked up stuff occurs inside ForwardBackward()
     -> occurs inside net.hpp::Forward() or before
     -> 


STEP 2
======

issue with the update. after 1 iteration, next loss is 14
for SBL, 73 for SL. bottom_diff takes on rubbish values.

so:
- bwd pass is wrong
- weight update is wrong

-> after 1 iteration, net only outputs 1s or 0s!
   -> so z = <x,w> + b can easily = 0 ? how?

-> ok, cost function seems to be working now. no nans or infs,
   and trianing error gets minimised.


   
STEP 3
======

Why is accuracy so weird?
-> find out whether same net loaded in by printing out param values
   -> have identical train and val files with just 128 cases
      compare output probs

Examine outputs
-> is this harsh error preventing the net from learning anything?
   -> ie all outputs are around 0.5, it's very confused
   -> find out by comparing benchmarks
      -> ground sheet outputs
      	 -> min class is 1, so other way around
      	 -> uh oh, forgot to get SL to print them out
      	 -> 23-08-2014 has them, 22-08-2014 doesnt
      	 -> 22-08-2014 is from old build, you can compare train time
      	    series with 23* to make sure new build isn't doing
	    anything different or wrong
      -> scrape zones outputs
      	 -> min class is 0
   -> if so, make it less harsh?
      -> less extreme renormalisation
      -> only penalise if output <=0.5 ie introduce kink in cost
      	 function
	 -> formula?
   -> implement under-sampling like the paper says

   
Test if correctly implemented:
-> graphic06: on a dataset of 6 images, perfectly balanced, batchsize 6, 
   train and val sets the same
   -> prob outputs not same for val as for train
      -> calling bottom[0] in SBL, prob_ in PCA
         is one of them wrong?
	 maybe SBL is wrong, hence bad results below?
   -> loss same for sbl and sl at iter_1, but not afterwards
   -> bottom_diff not same for sbl as for sl
   -> CAREFUL! after debugging, get back data/ground_sheet/temp

Read the paper threshold paper properly!   


IDENTIFIED PROBS & SOLS:
-> what if prior is (1,0)
-> must implement under- and over-sampling as well
   shit that will be hard



=====

Fuck it, that is too hard. And it might not even work. Threshold
works, is easier to implement, and probably has more powerful results.

currently:
-> graphic07 writing python wrappers for running the net
-> idea is to get the prediction probs, and assign flags based on them
& threshold
-> debugging run_classifier.py
   -> done

=====


Need to:
- train nets
  -> use optimal backprop freeze
- use Redbox data
  -> script to use Redbox data from a certain date
     -> graphic07 meta.zip
  -> try multiple threshold dates
  -> use best performing network so far
     -> clampdet 94%, what arch was that?
     	-> clampdet                           0.2 
	-> no_thresh-fine                     0.12
	-> thresh                             0.12
	-> thresh_freeze_backprop5/13         0.7
	-> thresh_freeze_backprop5/14         0.7
	-> thresh_freeze_backprop5.5/11       0.15
	-> thresh_freeze_backprop5.5/12       0.15
	-> thresh_freeze_backprop5.5/13       0.39
	-> thresh_freeze_backprop5.5/14       0.4     
	-> thresh_freeze_backprop5.5/15       0.18
	-> thresh_freeze_backprop6/11         0.17   
	-> thresh_freeze_backprop6/13         0.4   
	-> thresh_freeze_backprop7/11         0.17
     ok seems perf driven by:
     - expressiveness 
     - whether lr_policy fucked up
     still space for optimising both
	
     -> better than optimal backprop freeze?
- write up threshold
- write up sbl


=====


screw the Redbox data. focus on running experiments
from below.


-> ReLU maths
   -> neat writeup
   
-> Early stopping maths
   -> draft
   -> neat writeup
   
-> Generic clamp
   -> restructure
   -> neat writeup

-> Transfer learning
   -> freezing backprop
   -> initialising weights

-> Class imbalance
   -> under-sampling
   -> in-net threshold
   -> SBL
   -> test-time threshold

-> Final Results


=====

TRAIN

- Generic Clamp:
  -> mis-labelling, how to show?

- Transfer Learning
  -> test run
     -> with:                                 DONE
     -> w/out:                                DONE
  -> clampdet, freeze backprop on:
     -> none:  clampdet/none                  DONE
     -> conv1: clampdet/                      DONE
     -> conv2: clampdet/                      DONE
     -> conv3: clampdet/                      DONE
     -> conv4: clampdet/                      TODO
     -> conv5: clampdet/                      TODO
     -> fc6:   clampdet/                      TODO?
     -> fc7:   clampdet/                      DONE
  -> weight initialisation
     -> reinit: clampdet/none_reinit          DONE
     -> ¬reinit: clampdet/none                DONE
  -> parametric vs non parametric
     -> linear SVM: clampdet/linSVM           TODO
     -> best net fr above: clampdet/none?

Class Imbalance:
-> Test Run without TL
     clampdet/tl_wout                         DONE
     clampdetCI98/tl_wout                     DONE
-> Transfer Learning
     clampdetCI98/tl_wout                     DONE
     clampdetCI98/none_reinit_bs128_lr4       DONE
     clampdetCI98/none_bs128_lr4              DONE
-> Batch Size
     clampdetCI98/none_bs128_lr4              DONE
     clampdetCI98/none_bs256_lr4              DONE
-> Learning Rate
     clampdetCI98/none_bs256_lr4              DONE
     clampdetCI98/none_bs256_lr5              TRAINING
-> Examine Impact with run_classifier
     clampdet/none                            DONE
     clampdetCI(97)/none(_bs256_lr5)          DONE             
     clampdetCI98/none(_bs256_lr5)            DONE           
-> SBL
     clampdetCI98/none_bs256_lr4              DONE           
     clampdetCI98/none_bs256_lr4_sbl          TRAINING
     clampdetCI98/none_bs256_lr5_sbl          RETRAINING
     clampdetCI98/conv5_bs256_lr5_sbl         TRAINING
     clampdetCI98/fc7_bs256_lr5_sbl           DONE
      
Conserving Spatial Information:
-> Test Run
     soil_contam/tl_wout
     soil_contam/noneC_lr5_sbl                TRAINING
     note bs128 saturates gpu mem
-> Remove pooling and an fc layer
     soil_contam/nopool_sl                    TRAINING
-> SBL
     soil_contam/nopool                       TRAINING
     

Final Results:
-> transfer top notch clampdet net instead?
     clampdet/none_best                       DONE
     soil_contam/noneC                        DONE
     hatch_markings/noneC                     DONE
     hatch_markings/none                      TODO
-> pooling loses spatial informations
     soil_contam/poolless                     TODO
   
-> what is the best arch?
  -> do NOT reinit (not enough data, at least not with UnderSampling)
  -> clampdet
  -> ground sheet
  -> hatch markings
  -> insertion depth markings
  -> scrape zones
  -> joint misaligned
  -> contamination
  -> fitting proximity
  -> scraping peeling

=====

TRAIN OLD Class Imbalance:
-> Examine Impact:
   
-> Test Run 
     clampdetCI/BULLSHIT                      DONE
     clampdetCI/none                          TRAINING
-> Under-Sampling
     clampdetCI/tl_wout                       TODO    
     clampdetCI_us/tl_wout                    TODO       
     clampdet/none                            TRAINING
     clampdet_us/none                         DONE
-> Transfer Learning
     clampdet/tl_wout                         DONE
     clampdet/none                            TRAINING 
     clampdetCI_us0.5/none     (*)            TODO       
     clampdetCI_usAbove/none   (a)            TODO - need (*)
     clampdetCI_usBelow/none   (b)            TODO - need (*)
     ---
     if fail: {freezeBest}
     (clampdetCI_usAbove/fc{6or7}?)           TODO? - dep (a,b)
     (clampdetCI_usBelow/fc{6or7}?)           TODO? - dep (a,b)
-> Bayesian Cross Entropy
     clampdetCI{best_from_above}/sbl          TODO - dep trans l
-> Over-Sampling
     clampdetCI_os/none                       TODO
-> Test time Threshold
     clampdetCI{best_from_above}/thresh at target_min


=====

WRITE

What do I still need to write (from scratch)?
- Background:
  -> why neural nets so good?
     because they generalise so well
     why do we care about generalising well?
     because of curse of dimensionality
     bit.ly/1pEOuYV
     how does neural net generalise so well?
     with distributed representation
     ie hierarchical representation
     ie compositionality of parameters
     ie exponential compactness

  -> grad descent polynomial approximation
     
  -> AlexNet in detail: stoch pooling paper
     not Rob Fergus tutorial! too long

- Justify independent binary classifiers
  
- SBL


=====

ANALYSE:
  -> run_classifier.py on them
  -> plots
  -> write up:
     -> comments
     -> plot
     -> table from run_classifier

DONE when final plots & rough comments present
TODO otherwise

Transfer Learning:
-> Test Run                                       TODO
-> Freeze Backprop                                TODO
     clampdet/conv1
     clampdet/conv2
     clampdet/conv3
     clampdet/conv4
     clampdet/conv5
     clampdet/fc6
     clampdet/fc7
-> Reinit Weights                                 TODO
     clampdet/none_reinit
     clampdet/none
-> Parametric vs Non-Parametric                   TODO
     clampdet/linSVM
     clampdet/none ? (best so far)

Class Imbalance:
-> Test Run without TL
     clampdet/tl_wout
     clampdetCI98/tl_wout
-> Transfer Learning
     clampdetCI98/tl_wout
     clampdetCI98/none_reinit_bs128_lr4
     clampdetCI98/none_bs128_lr4
-> Batch Size
     clampdetCI98/none_bs128_lr4
     clampdetCI98/none_bs256_lr4
-> Learning Rate
     clampdetCI98/none_bs256_lr4
     clampdetCI98/none(_bs256_lr5)
-> Examine Impact with run_classifier
     clampdet/none
     clampdetCI(97)/none(_bs256_lr5)
     clampdetCI98/none(_bs256_lr5)

-> SBL
     talk about choosing appropriate backprop
     ground_sheet_sbl/logs/pcba/GettingThere has trace
     
Maybe:     
-> Under-Sampling
-> Transfer Learning
-> Bayesian Cross Entropy
-> Over-Sampling
-> Test time Threshold
     clampdetCI/tl_wout       - benchmark                       
     clampdetCI{best_from_above}/thresh at target_min

Final Results:
-> transfer top notch clampdet instead?
     clampdet_fine_train_iter_{best}
-> transfer top notch clampdet net instead?
     clampdet/none_best                       TODO
     soil_contam/noneC                        TODO
     hatch_markings/noneC                     TODO
     hatch_markings/none                      TODO
-> pooling loses spatial information
     soil_contam/poolless                     TODO
   
     
======

ANALYSE OLD class imbalance

-> Test Run / batchSize                           TODO
     clampdetCI/BULLSHIT
     clampdetCI/none
-> Under-Sampling
     clampdetCI/tl_wout                             
     clampdetCI_us/tl_wout                             
     ---
     plot_clampdet_none                             
     plot_clapdet_us_none                             
-> Transfer Learning
     clampdetCI/tl_wout                             
     clampdetCI/none                             
     ---
     clampdetCI_us0.5/none                             
     ---
     : us{Best}
     clampdetCI/tl_wout       - benchmark                       
     clampdetCI_usAbove/none                             
     clampdetCI_usBelow/none                             
     ---
     if fail: {freezeBest}
     (clampdetCI_usAbove/fc{6or7}?)                             
     (clampdetCI_usBelow/fc{6or7}?)                             
-> Bayesian Cross Entropy
     clampdetCI/tl_wout       - benchmark                       
     clampdetCI{best_from_above}
     clampdetCI{best_from_above}/sbl
-> Over-Sampling
     with clampdet you didnt try no reinit, do so now:
     clampdetCI/tl_wout       - benchmark                       
     clampdetCI{best_from_above}
     clampdetCI_os/none
     ---
     if clampdetCI_os/none better than clampdetCI_us0.5/none:
     clampdetCI_os/{freezeBest} ADD TO TODO LIST!
-> Test time Threshold
     clampdetCI/tl_wout       - benchmark                       
     clampdetCI{best_from_above}/thresh at target_min

======


Next:

-> prepare all class imbalance prototxts
-> run them

-> transfer 
-> ANALYSE transfer learning TODOs

- break -

-> check what has finished training
-> save solverstates in directories!
-> update TRAIN statuses
-> ANALYSE class imbalance TODOs


====

Ok, just realised even CI98 doesn't get bad min
- but maybe that's thanks to tl, no reinit, large batch, small lr

new path:
- examine impact of class imbalance
  -> different imbalance rates with nest arch until now
     -> batchsize 128
     -> lr 0.0001
  
- cure class imbalance:
     
  
=====

SHORT TERM

-> figure out which clampdet/none iter is best
-> delete the others
-> is best iter as alternative transfer model
   -> soil_contam/none
   -> soil_contam/noneC  # means transfer from clampdet task

-> get evidence for sbl intuition
   -> clampdetCI98/none_bs256_lr4 (already trained?)
   -> clampdetCI98/none_bs256_lr4_sbl (multi snapshots cos dunno
      when to early stop cos inadequate val err)
   -> run_classifier to see whether perf on positives better

-> benchmark SBL:
   -> run_classifier threshold that maximises pca
   -> under sampling:
      need to remove min class to get target imbalance
      and then under smaple to get target bad min

-> contam poolless      
   that's for another time
   
