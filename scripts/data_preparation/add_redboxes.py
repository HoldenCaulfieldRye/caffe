#!/usr/bin/env python
import os, random, shutil
from os.path import join as oj
import cPickle as pickle
import subprocess, random
import setup


# why separate from bring in negatives?
# NOT RANDOM! USING TAIL
# imbalance_multiple is more than by how much maj class is bigger than min class in redbox. it's a heuristic to speed up computation
def bring_redbox_positives(task, flag, add_num, imbalance_multiple):
  here = os.getcwd()
  os.chdir('/data/ad6813/caffe/data/'+task)
  cmd = "find /data/ad6813/pipe-data/Redbox/raw_data/dump/ -name '*.dat' | tail -"+str(add_num*imbalance_multiple)+" | xargs -i grep -l '"+flag+"' {} | tail -"+str(add_num)+" | cut -d'.' -f 1 | xargs -i echo '{}.jpg 1' >> train.txt"
  # print cmd
  p = subprocess.Popen(cmd, shell=True)
  p.wait()
  shuffle_file('train.txt')
  os.chdir(here)


def shuffle_file(fname):
  contents = open(fname, 'r').readlines()
  random.shuffle(contents)
  open(fname, 'w').writelines(contents)

 
# integrate bring in positives into this? 
def bring_redbox_negatives(task, avoid_flags, add_num, pickle_fname, data_dir, fn_train, using_pickle):
  neg_classification = ' 0' # see dump_to_files [('Default',0),(task,1)]
  if os.path.isfile(oj(os.getcwd(),'redbox_vacant_'+task+'_negatives.pickle')) and using_pickle:
    print "Found pickle dump of vacant, non-perfect Redbox images without %s flag. Using it."%(task)
    notperf = pickle.load(open(pickle_fname,'r'))

  else:
    notperf, total = [], []
    for fname in os.listdir(data_dir):
      if fname.endswith('.dat'):
        total.append(fname)

    print 'Gathering vacant, non-perfect Redbox images without %s flag...'%(task)
    count = 0
    with open(fn_train,'r') as f_already:
      c_already = f_already.readlines()
      c_already = [line.split(' ')[0] for line in c_already]
      for i in range(len(total)):
        content = open(oj(data_dir,total[i]),'r').readlines()
        content = [line.strip() for line in content]
        if all([len(content) > 0,
                len([flag for flag in content if flag in avoid_flags])==0,
                total[i] not in c_already]):
          notperf.append(data_dir+total[i][:-4]+'.jpg'+neg_classification+'\n')
          count += 1
          if count > add_num: break

    random.shuffle(notperf)
    print "Gathering completed."

  print "Adding %i of them to %s"%(add_num,fn_train)
  newcomers, notperf_left = notperf[:add_num], notperf[add_num:]
  pickle.dump(notperf_left, open(pickle_fname,'w'))
  print "%s updated"%(pickle_fname)

  with open(fn_train,'r') as f_train:
    c_train = f_train.readlines()
  c_train += newcomers
  random.shuffle(c_train)
  f_train = open(fn_train,'w')
  # print "writing:", c_train
  f_train.writelines(c_train)


def blur_no_infogain(blue_c_imb, data_dir, task, pos_class):
  # assumed b_imb == 0.5
  blue_dir = data_dir.replace('Red','Blue')
  d_red = setup.get_label_dict_knowing(data_dir, task, pos_class)
  d_blue = setup.get_label_dict_knowing(blue_dir, task, pos_class)
  red_c_imb=float(len(d['Default']))/(len(d[task])+len(d['Default']))
  if red_c_imb >= blue_c_imb:
    # can't add all negatives
    num_pos = len(d_red[task])
    num_neg = num_neg * (len(d_blue['Default'])/len(d_blue[task]))
  else:
    num_neg = len(d_red['Default'])
    num_pos = num_pos * (len(d_blue[task])/len(d_blue['Default']))
  return num_pos, num_neg
  
  
def what_redbox_numbers(c_imb, b_imb, data_dir, task, pos_class,
                        b_pos, b_neg):
  d = setup.get_label_dict_knowing(data_dir, task, pos_class)
  red_c_imb=float(len(d['Default']))/(len(d[task])+len(d['Default']))
  if red_c_imb >= c_imb:
    r_pos = len(d[task])
    r_neg = r_pos * (b_imb/(1-b_imb)) * (r_pos/(r_pos))
    return r_pos, r_neg
    # if red_c_imb lower, would have to 
  elif red_c_imb <= c_imb:
    print "class imbalance going to decrease! :D"
    return len(d[task]), len(d['Default'])*(c_imb/red_c_imb)
  

def same_amount_as_bluebox(data_dir, task, pos_class):
  d = setup.get_label_dict_knowing(data_dir, task, pos_class)
  # ASSUMING MODEL LEARNS P(label|data) !
  return len(d[task]), len(d[task])
  # return len(d[task]), len(d['Default'])
  ## that would assume need to keep Redbox imbalance == Blue imbalance
  ## which one is true??


def delete_some_files(fname, del_num):
  count, new = 0, []
  content = open(fname, 'r').readlines()
  for line in content:
    if line.endswith('0\n') and count < del_num: count += 1
    else: new.append(line+'\n')
  open(fname,'w').writelines(new)
  

if __name__ == '__main__':
  import sys

  task = 'scrape'
  avoid_flags = ['NoVisibleEvidenceOfScrapingOrPeeling','PhotoDoesNotShowEnoughOfScrapeZones']
  using_pickle = False
  pickle_fname = 'redbox_vacant_'+task+'_negatives.pickle'
  data_dir = '/data/ad6813/pipe-data/Redbox/raw_data/dump/'
  fn_train = '/data/ad6813/caffe/data/scrape/train.txt'
  imbalance_multiple = 10
  
  add_num = 20000

  bring_redbox_negatives(task, avoid_flags, add_num, pickle_fname, data_dir, fn_train, using_pickle)

  flag = 'NoVisibleEvidenceOfScrapingOrPeeling'
  print 'bringing in redbox positives...'  
  bring_redbox_positives(task, flag, add_num, imbalance_multiple)












