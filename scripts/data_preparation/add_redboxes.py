#!/usr/bin/env python
import os, random, shutil
from os.path import join as oj
import subprocess, random
import setup


# why separate from bring in negatives?
# NOT RANDOM! USING TAIL
# imbalance_multiple is more than by how much maj class is bigger than min class in redbox. it's a heuristic to speed up computation
def bring_redbox_positives(task, flags, add_num, redbox_dir, fn_train):
  added = []
  listdir = os.listdir(redbox_dir)
  random.shuffle(listdir)
  for fl in listdir:
    if fl.endswith('.dat'):
      pres = False
      with open(oj(redbox_dir,fl), 'r') as f:
        for line in f:
          if line.strip() in flags:
            pres = True
            break
      if pres:
        added.append(fl)
        if len(added) >= add_num:
          break

  with open(fn_train, 'a') as f:
    for fl in added:
      fl = fl.replace('dat','jpg')
      f.write("\n"+oj(redbox_dir,fl)+ " 1")
  

def shuffle_file(fname):
  num_pos, num_neg = 0, 0
  contents = open(fname, 'r').readlines()
  for line in contents:
    if line.endswith('1\n'): num_pos += 1
    elif line.endswith('0\n'): num_neg += 1
  random.shuffle(contents)
  open(fname, 'w').writelines(contents)
  # print "there are now %i positives"
  return num_pos, num_neg

 
# integrate bring in positives into this? 
def bring_redbox_negatives(task, avoid_flags, add_num, data_dir, fn_train):
  neg_classification = ' 0' # see dump_to_files [('Default',0),(task,1)]
  notperf, total = [], []
  for fname in os.listdir(data_dir):
    if fname.endswith('.dat'):
      total.append(fname)

  print "Gathering vacant Redbox images without %s's flags..."%(task)
  count = 0
  with open(fn_train,'r') as f_already:
    c_already = f_already.readlines()
    c_already = [line.split(' ')[0] for line in c_already]
    for i in range(len(total)):
      content = open(oj(data_dir,total[i]),'r').readlines()
      content = [line.strip() for line in content]
      if all([len([flag for flag in content if flag in avoid_flags])==0,
              total[i] not in c_already]):
        notperf.append(oj(data_dir,total[i][:-4])+'.jpg'+neg_classification+'\n')
        count += 1
        if count > add_num: break

  random.shuffle(notperf)
  print "Gathering completed."

  print "Adding %i negatives to %s"%(add_num,fn_train)
  newcomers, notperf_left = notperf[:add_num], notperf[add_num:]

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
  red_c_imb=float(len(d_red['Default']))/(len(d_red[task])+len(d_red['Default']))
  blue_c_imb = float(blue_c_imb)
  if red_c_imb >= blue_c_imb:
    # can't add all negatives
    num_pos = len(d_red[task])
    num_neg = num_pos * (blue_c_imb/float(1-blue_c_imb))
    # num_neg = num_pos * (len(d_blue['Default'])/len(d_blue[task]))
  else:
    # can't add all positives
    k = ((1-blue_c_imb)/float(blue_c_imb))
    print k
    for key in d_red.keys():
      try:
        print len(d_red[key]), key
      except: pass
    num_neg = len(d_red['Default'])
    num_pos = num_neg * k
    # num_pos = num_neg * (len(d_blue[task])/len(d_blue['Default']))
  return int(num_pos), int(num_neg)
  
  
def what_redbox_numbers(c_imb, b_imb, data_dir, task, pos_class,
                        b_pos, b_neg):
  # big prob: after redbox sampling, imbalance has changed.
  # so actually, redbox sampling and undersampling both need to be
  # determined before either takes place.
  # other prob: given b_imb, compute num_neg num_pos
  # maybe easier is given info gain, compute num_neg num_pos
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
  data_dir = '/data/ad6813/pipe-data/Redbox/'
  fn_train = '/data/ad6813/caffe/data/scrape/train.txt'
  imbalance_multiple = 10
  
  add_num = 20000

  bring_redbox_negatives(task, avoid_flags, add_num, data_dir, fn_train)

  flag = 'NoVisibleEvidenceOfScrapingOrPeeling'
  print "Adding %i positives to %s..."%(add_num,fn_train)
  print task, flag, add_num, imbalance_multiple
  bring_redbox_positives(task, flag, add_num, imbalance_multiple)

