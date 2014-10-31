import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys, shutil
import yaml
from os.path import join as oj
import subprocess
sys.path.append(os.path.abspath('../plot_data'))
import plot
sys.path.append(os.path.abspath('../data_preparation'))
import setup_all as sa
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../python'))
import caffe, check
from caffe.proto import caffe_pb2
from create_deploy_prototxt import *

caffe_root = '../../'  # this file is expected to be in {caffe_root}/exampless
sys.path.insert(0, caffe_root + 'python')

# usage:

# python run_classifier.py classifier-dir=/homes/ad6813/net-saves/clampdet/linSVM symlink-dir=../../data/clampdet data-info=../../data_info/clampdet --redbox

# python run_classifier.py classifier-dir=/homes/ad6813/net-saves/clampdet/none symlink-dir=../../data/clampdet data-info=../../data_info/clampdet

# flags:
  # --mislab
  # --redbox

# for redbox mode:
#   data-info needs a 'redbox' dir of classifications for all RB imgs
#   data-dir needs a 'redbox' dir of symlinks to all rdbox images

REDBOX_DIR = '/data/ad6813/pipe-data/small' #Redbox'

# Note! data-dir should be data/<name>, not data/<name>/test

def classify_data(classifier_dir, symlink_dir, data_info, PRETRAINED, redbox=False):
  N = 96
  classifier_name = classifier_dir.split('/')[-1]  
  if classifier_name.split('-fine')[0]+'_deploy.prototxt' not in os.listdir(classifier_dir):
    create_deploy_file(classifier_dir)
    
  MODEL_FILE = oj(classifier_dir, classifier_name.split('-fine')[0]+'_deploy.prototxt')
  MEAN_FILE = np.load(get_np_mean_fname(symlink_dir))
  print 'loading network...'
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                         image_dims=(256, 256), input_scale=255,
                         mean=MEAN_FILE, channel_swap=(2,1,0))
  # flow of control:
  #   classifier::__init__(
  #   classifier::caffe.Net.__init__()
  print 'network loaded successfully'
  # set phase to test since we are doing testing
  net.set_phase_test()
  net.set_mode_gpu()
  d = {'fname': [],
       'pred': [],
       'time': [],
       'dude': [],
       'label': [],
       'pred_lab_thresh': [],
       'pred_lab_std': [],
       'pot_mislab': []}
  # load images
  if redbox:
    imgs, d = load_all_images_from_dir(d, oj(symlink_dir,'redbox'), redbox)
  else:
    imgs, d =  load_all_images_from_dir(d, oj(symlink_dir,'test'))

  # classify images
  num_imgs = len(d['fname'])
  print "computing preds..."
  d['pred'] = net.predict(imgs[:N])
  # print pred
  if num_imgs > N:
    for i in range(1,num_imgs/N):
      d['pred'] = np.append(d['pred'],net.predict(imgs[i*N:(i+1)*N]),axis=0)
    d['pred']=np.append(d['pred'],net.predict(imgs[-(len(imgs)%N):]),axis=0)
  print "preds computed."

  # save preds
  assert len(d['pred']) == num_imgs
  np.save(oj(data_info, PRETRAINED.split('/')[-1]+'_pred.npy'), d)
  return d

def augment_read(data_info):
  here = os.getcwd()
  shutil.copy('/data/ad6813/caffe/scripts/run_classifier/augment_read.sh', data_info)
  os.chdir(data_info)
  subprocess.call('chmod +x augment_read.sh', shell=True)
  subprocess.call('./augment_read.sh', shell=True)
  os.remove('augment_read.sh')
  os.chdir(here)
  
def get_flag_and_thresh(data_info):
  ''' flag_val is the number in data_info/[model]/read.txt which indexes
  the class corresponding to when flag is present. ''' 
  flag_val, thresh = 0, 0.5
  rl = open(oj(data_info,'read.txt'),'r').readlines()
  
  if len([l for l in rl if 'flag_val' in l]) == 0:
    # set up read.txt to contain flag val and threshold
    augment_read(data_info)
    rl = open(oj(data_info,'read.txt'),'r').readlines()
    
  rl = [l.split() for l in rl]
  for l in rl[2:]:
    if l == ['1','flag_val']: flag_val = 1
    elif l[1] == 'threshold': thresh = float(l[0])
  # if got no thresh to return, means read.txt needs be filled in
  return flag_val, thresh

def get_pretrained_model(classifier_dir):
  suggest = os.listdir(classifier_dir)
  suggest = [fname for fname in suggest
             if 'iter' in fname and 'solverstate' not in fname]
  if len(suggest) > 1:
    for elem in enumerate(suggest): print elem
    idx = int(raw_input("\nWhich model? "))
    return oj(classifier_dir,suggest[idx])
  elif len(suggest) == 1:
    return oj(classifier_dir,suggest[0])
  else:
    print "ERROR: no model found in", classifier_dir
    exit()


def create_deploy_file(classifier_dir):
  train_file = get_train_file(classifier_dir)
  content = train_file.readlines()
  content = edit_train_content_for_deploy(content)
  write_content_to_deploy_file(classifier_dir, content)
  

def get_np_mean_fname(symlink_dir):
  proto_img_fname = ''
  for fname in os.listdir(symlink_dir):
    if fname.endswith('mean.binaryproto'):
      print 'found binaryproto: %s'%(fname)
      proto_img_fname = fname
      break
  if proto_img_fname == '':
    print 'ERROR: no *mean.npy nor *mean.binaryproto found in %s'%(symlink_dir)
    sys.exit()
  # er wait how does it know where the proto img file is?
  blob = caffe_pb2.BlobProto()
  data = open(oj(symlink_dir,proto_img_fname), "rb").read()
  blob.ParseFromString(data)
  nparray = caffe.io.blobproto_to_array(blob)[0]
  npy_mean_fname = (proto_img_fname.split('_mean.binaryproto')[0]).split('_fine')[0]+'_mean2.npy'
  npy_mean_file = file(oj(symlink_dir,npy_mean_fname),"wb")
  np.save(npy_mean_file, nparray)
  npy_mean_file.close()  
  return oj(symlink_dir, npy_mean_fname)

          
def load_all_images_from_dir(d, test_dir, redbox=False):
  imgs = []
  d['fname'] = os.listdir(test_dir)
  print 'loading images from %s...'%(test_dir)
  # d_multJoints is a dict: fname -> joint_name
  d_multJoints = create_dict_jname(REDBOX_DIR)
  for fname in d['fname']:
    full_fname = oj(test_dir, fname)
    imgs.append(caffe.io.load_image(full_fname))
    if redbox:
      [dude,time] = get_(REDBOX_DIR,fname,['InspectedTime','InspectedBy'])
      l_time = time.split('/')
      time = l_time[2] + '-' + l_time[1] + '-' + l_time[0]
      d['time'].append(time)
      d['dude'].append(dude)
  print 'finished loading images.'
  return imgs, d

def create_dict_jname(data_dir):
  file_multJoints = '/data/ad6813/pipe-data/Redbox/multJoints.txt'
  multJoints = {}
  if not os.path.exists(file_multJoints):
    cmd = "sshpass -p '9northerncighT!' scp graphic06.doc.ic.ac.uk:"+file_multJoints+' '+file_multJoints
    print 'about to run', cmd
    subprocess.call(cmd, shell=True, stdout = subprocess.PIPE, stderr=subprocess.STDOUT)
  for line in open(file_multJoints,'r').readlines():
    for img in line.split()[1:]:
      multJoints[img+'.jpg'] = line.split()[0]
  return multJoints
  
def get_(data_dir, fname, what):
  ret = []
  meta_name = fname.split('.')[0] + '.met'
  data_dir = REDBOX_DIR
  for line in open(oj(data_dir,meta_name),'r').readlines():
    for field in what:
      if line.startswith(field):
        ret.append(line.split(field+'=')[-1].split()[0][:10])
  return ret
  
                 
def compute_classification_stats(d, data_info, redbox=False):
  # this comes early because flag_val prompts user
  flag_val, threshold = get_flag_and_thresh(data_info)
  num_imgs = len(d['fname'])
  # get data_info test file
  if redbox:
    print 'opening redbox data info'
    lines = open(oj(data_info,'redbox.txt'),'r').readlines()
  else:
    lines = open(oj(data_info,'test.txt'),'r').readlines()
  _class = {}
  for line in lines: _class[line.split()[0]] = line.split()[1]
  # print "lines:", lines
  try:
    assert set(d['fname']) == set(_class.keys())
  except:
    print "don't match: d['fname']", d['fname']
    print "and _class.keys()", _class.keys()
    exit
  # fill with true labels
  d['label'] = [int(_class[el]) for el in d['fname']]
  # fill in predicted labels and flag if potentially mislab
  # *_thresh is with classification boundary according to threshold
  # *_std is with classification boundary at 0.5
  false_pos_thresh, num_pos, false_neg_thresh, num_neg, false_neg_std, false_pos_std = 0, 0, 0, 0, 0, 0
  # print "\nd['label'] has types %s and flag_val of type %s \n"%(type(d['label'][0]),type(flag_val))
  for idx in range(num_imgs):
    if d['label'][idx] == flag_val:
      print "%s is a positive, with preds %s"%(d['fname'][idx])
      print "its preds are", d['pred'][idx]
      num_pos += 1
    else: num_neg += 1
    # assign predicted label wrt threshold
    if d['pred'][idx][flag_val] >= threshold:
      d['pred_lab_thresh'].append(flag_val) 
      # print "thresh thinks no clamp! appending", flag_val
    else:
      d['pred_lab_thresh'].append((flag_val+1)%2)
      print "thresh thinks clamp! appending", (flag_val+1)%2
    # assign predicted label in std way
    if d['pred'][idx][flag_val] >= 0.5:
      d['pred_lab_std'].append(flag_val) 
      # print "std thinks no clamp! appending", flag_val, "\n"
    else:
      d['pred_lab_std'].append((flag_val+1)%2)
      print "std thinks clamp! appending", (flag_val+1)%2, "\n"
    # correct thresh classification or not 
    if d['pred_lab_thresh'][idx] != d['label'][idx]:
      if d['label'][idx] == flag_val: false_neg_thresh += 1
      else: false_pos_thresh += 1
    # correct std classification or not 
    if d['pred_lab_std'][idx] != d['label'][idx]:
      d['pot_mislab'].append(idx)
      if d['label'][idx] == flag_val: false_neg_std += 1
      else: false_pos_std += 1

  print 'false_neg_thresh: %i, false_pos_thresh: %i'%(false_neg_thresh,false_pos_thresh)
  print 'false_neg_std: %i, false_pos_std: %i'%(false_neg_std,false_pos_std)
  print 'num_neg: %i, num_pos: %i'%(num_neg,num_pos)
  # compute accuracies
  d['accuracy']= {}
  d['accuracy']['total_thresh'] = 1-(false_neg_thresh+false_pos_thresh)/float(num_imgs)
  d['accuracy']['pos_thresh'] = 1-false_neg_thresh/float(num_pos)
  d['accuracy']['neg_thresh'] = 1-false_pos_thresh/float(num_neg)
  d['accuracy']['total_std'] = 1-(false_neg_std+false_pos_std)/float(num_imgs)
  d['accuracy']['pos_std'] = 1-false_neg_std/float(num_pos)
  d['accuracy']['neg_std'] = 1-false_pos_std/float(num_neg)
  print "d['accuracy']", d['accuracy']
  return d


def compute_kpi(d):
  num_imgs = len(d['fname'])
  flag = get_flag_and_thresh(data_info)[0]
  # create array (idx,prob(pos)) of all positives
  pos = [(idx,float(d['pred'][idx][flag]))
         for idx in range(num_imgs)
         if d['label'][idx] == flag]
  print 'check same with above! num_pos:', len(pos)
  # sort array descending prob(pos)
  pos = sorted(pos, key=lambda x: x[1], reverse=True)
  # Sig_level is prob(pos) for i-th entry where float(i/len) = 0.95
  print 'sig levels required for following accuracy on positives:'
  print '70\%:',pos[int(0.7*len(pos))][1]
  print '80\%:',pos[int(0.8*len(pos))][1]
  print '90\%:',pos[int(0.9*len(pos))][1]
  print '95\%:',pos[int(0.95*len(pos))][1]
  Sig_level = float(pos[int(0.95*len(pos))][1])
  automated = [idx for idx in range(num_imgs)
               if float(d['pred'][idx][flag]) < Sig_level]
  return Sig_level, len(automated)/float(num_imgs)

def save_mislabs(d, data_info, PRETRAINED):
  mislab_dir = oj(data_info,'potential_mislabels_'+PRETRAINED.split('/')[-1])
  try: os.mkdir(mislab_dir)
  except:
    shutil.rmtree(mislab_dir)
    os.mkdir(mislab_dir)
  for idx in d['pot_mislab']:
    shutil.copy(oj(data_info,'test',d['fname'][idx]), mislab_dir)
  print "saving potential mislabels to %s"%(mislab_dir)


def print_classification_stats(d):
  print 'with threshold at test only:'
  print 'accuracy overall: ', d['accuracy']['total_thresh']
  print 'accuracy on positives: ', d['accuracy']['pos_thresh']
  print 'accuracy on negatives: ', d['accuracy']['neg_thresh']
  print 'with standard 0.5 classification:'
  print 'accuracy overall: ', d['accuracy']['total_std']
  print 'accuracy on positives: ', d['accuracy']['pos_std']
  print 'accuracy on negatives: ', d['accuracy']['neg_std']
  # find highest sig_level that raises >=95% of true positives,
  # and compute % workload that is automated
  Sig_level, pct_auto = compute_kpi(d)
  print 'sig level required for 95% accuracy on positives:',Sig_level
  print 'this enables', pct_auto, 'automation'

def create_redbox_data_info_etc(symlink_dir, data_info):
  data_dir = REDBOX_DIR
  All = sa.get_label_dict(data_dir)
  total_num_images = All.pop('total_num_images')
  Keep = sa.classes_to_learn(All)
  Keep = sa.default_class(All, Keep)
  total_num_check = sum([len(Keep[key]) for key in Keep.keys()])
  if total_num_images != total_num_check:
    print "\nWARNING! started off with %i images, now have %i distinct training cases"%(total_num_images, total_num_check)
  if len(Keep.keys()) > 2:
    Keep,num_output = sa.merge_classes(Keep)
    Keep,num_output = sa.check_mutual_exclusion(Keep, num_output)
  dump = symlink_redbox_dataset(Keep,data_dir,oj(symlink_dir,'redbox'))
  dump_redbox_to_files(Keep, dump, data_info)
  for (f,num) in dump:
    if f == Keep['Default'][0]: return num

def symlink_redbox_dataset(Keep, from_dir, to_dir):
  if os.path.isdir(to_dir): shutil.rmtree(to_dir)
  os.mkdir(to_dir)
  dump = []
  for [num,key] in enumerate(Keep.keys()):
    dump += [[f,num] for f in Keep[key]]
  for i in xrange(len(dump)):
    if os.path.islink(ojoin(to_dir,dump[i][0])):
      old = dump[i][0]
      while os.path.islink(ojoin(to_dir,dump[i][0])):
        print '%s symlinked already, creating duplicate'%(dump[i][0])
        dump[i][0] = dump[i][0].split('.')[0]+'_.jpg'
      os.symlink(ojoin(from_dir,old),
                 ojoin(to_dir,dump[i][0]))
    else: os.symlink(ojoin(from_dir,dump[i][0]),
                     ojoin(to_dir,dump[i][0]))
  return dump
                           
def dump_redbox_to_files(Keep, dump, data_info):
  dfile = open(ojoin(data_info,'redbox.txt'),'w')
  dfile.writelines(["%s %i\n" % (f,num) for (f,num) in dump])
  dfile.close()

def arrange_preds_with_flag_val(d, flag_val):
  print '''WARNING! aligning preds with flag val like this only 
  works with balanced datasets. '''
  if any([sum(d['pred'][:,1])>sum(d['pred'][:,0]) and flag_val==1,
          sum(d['pred'][:,0])>sum(d['pred'][:,1]) and flag_val==0]):
    print "\nflipping pred columns around!\n"
    d['pred'] = np.fliplr(d['pred'])
  else: print "\nnot flipping pred columns around\n"

  return d

def plot_for_redbox(d, save_dir):
  plot_time(d, save_dir)
  plot_dudes(d, save_dir)

def plot_time(d, save_dir):
  if len(d['time']) != len(d['error']):
    print "len(d['time']) %i != %i len(d['error'])"%(len(d['time']),len(d['error']))
    sys.exit()
  # order by time
  data = np.array(zip(d['time'],d['error']), dtype=object)
  data = numpy.sort(data, axis=0)
  plt.ylim([0,1.2])
  x, y = data[:,0], data[:,1]
  plt.plot(x, y)
  plt.legend(loc='upper left')
  plt.xlabel('Inspected Time')
  plt.ylabel('Classification Error')
  # plt.title('Go on choose one')
  plt.grid(True)
  plt.savefig(oj(save_dir,'plot_redbox_'+save_dir.split('/')[-3]+'_'+save_dir.split('/')[-1]+'_time.png'))
  
def plot_dudes(d, save_dir):
  # get a 2d array of dudes of freq pot mislab
  data = {}
  num_imgs = len(d['dude'])
  for s in set(d['dude']): data[s] = 0
  for idx in d['pot_mislab']:
    data[d['dude'][idx]] += 1
  data = [[key, float(d[key])/num_imgs] for key in data.keys()]
  data = np.array(zip())
  fig = plt.figure()
  width = .35
  ind = np.arange(len(data))
  plt.bar(ind, data[:,1])
  plt.xticks(ind + width / 2, data[:,0])
  fig.autofmt_xdate()
  plt.xlabel('Inspected By')
  plt.ylabel('% mis-classifications')
  plt.savefig(oj(save_dir,'plot_redbox_'+save_dir.split('/')[-3]+'_'+save_dir.split('/')[-1]+'_dude.png'))



if __name__ == '__main__':
  print 'Warning: make sure that caffe is on the python path!'
  for arg in sys.argv:
    if "classifier-dir=" in arg:
      classifier_dir = os.path.abspath(arg.split('=')[-1])
    elif "symlink-dir=" in arg:
      symlink_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-info=" in arg:
      data_info = os.path.abspath(arg.split('=')[-1])

  redbox = False
  if '--redbox' in sys.argv: redbox = True
  
  if check.check(symlink_dir, data_info) != [0,0] and not redbox:
    print 'ERROR: mismatch between test files in data_dir and data_info'
    sys.exit()

  if redbox:
    flag_val = create_redbox_data_info_etc(symlink_dir, data_info)

  PRETRAINED = get_pretrained_model(classifier_dir)
  already_pred = oj(data_info, PRETRAINED.split('/')[-1]+'_pred.npy')
  if os.path.isfile(already_pred) and raw_input('found %s; use? ([Y]/N) '%(already_pred)) != 'N':
    d = (np.load(already_pred)).item()
  else:
    d = classify_data(classifier_dir, symlink_dir, data_info, PRETRAINED, redbox)

  if redbox:
    d = arrange_preds_with_flag_val(d, flag_val)
  
  # this should go in main as well?
  # get true labels, assign predicted labels, get metrics
  d = compute_classification_stats(d, data_info, redbox)
  print_classification_stats(d)
  
  # potential mislabels
  if "--mislab" in sys.argv:
    save_mislabs(d, data_info, PRETRAINED)

  # redbox plots
  if redbox:
    plot_for_redbox(d, data_info)
    
  
  # for faster prediction, turn off oversampling BUT!
  # you need to set oversampling in edit_train_content_for_deploy to
  # False... so you should probs merge the script into this one
  
  # To fully utilize the power of GPUs, you really want to:

  # Use larger batches, and minimize python call and data transfer overheads.
  # Pipeline data load operations, like using a subprocess.
  # Code in C++. A little inconvenient, but maybe worth it if your dataset is really, really large.
  

