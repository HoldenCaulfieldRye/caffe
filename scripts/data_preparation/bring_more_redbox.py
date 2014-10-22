#!/usr/bin/env python

import os, random, shutil
from os.path import join as oj
import cPickle as pickle

if __name__ == '__main__':

  task = 'scrape'
  classification = ' 0'
  using_pickle = True
  add_num = 20000 # how many imgs to add
  pickle_fname = 'redbox_vacant_'+task+'_negatives.pickle'
  redbox = '/data2/ad6813/pipe-data/Redbox/raw_data/dump/'
  fn_train = '/data2/ad6813/caffe/data/scrape/train.txt'

  if os.path.isfile(oj(os.getcwd(),'redbox_vacant_'+task+'_negatives.pickle')) and using_pickle:
    print "Found pickle dump of vacant, non-perfect Redbox images without %s flag. Using it."%(task)
    notperf = pickle.load(open(pickle_fname,'r'))

  else:
    notperf, total = [], []
    for fname in os.listdir(redbox):
      if fname.endswith('.dat'):
        total.append(fname)

    print 'Gathering vacant, non-perfect Redbox images without %s flag...'%(task)
    with open(fn_train,'r') as f_already:
      c_already = f_already.readlines()
      c_already = [line.split(' ')[0] for line in c_already]
      for i in range(len(total)):
        content = open(oj(redbox,total[i]),'r').readlines()
        if len(content) > 0:
          if total[i] not in c_already:
            notperf.append(redbox+total[i][:-4]+'.jpg'+classification+'\n')

    random.shuffle(notperf)
    print "Gathering completed."

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















