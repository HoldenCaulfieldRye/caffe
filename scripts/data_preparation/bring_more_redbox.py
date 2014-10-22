#!/usr/bin/env python

import os, random, shutil
from os.path import join as oj
import cPickle as pickle

if __name__ == '__main__':

  task = 'scrape'
  classification = ' 0'
  using_pickle = True
  pickle_fname = 'redbox_vacant_'+task+'_negatives.pickle'
  redbox = '/data2/ad6813/pipe-data/Redbox/raw_data/dump/'
  fn_train = '/data2/ad6813/caffe/data/scrape/train.txt'

  if os.path.isfile(oj(os.getcwd(),'redbox_vacant_'+task+'_negatives.pickle')) and using_pickle:
    print "Found pickle dump of vacant, non-perfect Redbox images without %s flag. Using it."%(task)
    notperf = pickle.load(pickle_fname,'rb')

  else:
    notperf, total = [], []
    for fname in os.listdir(redbox):
      if fname.endswith('.dat'):
        total.append(fname)

    print 'Gathering vacant, non-perfect Redbox images without %s flag...'%(task)
    with open(fn_train,'r') as f_already:
      c_already = f_already.readlines()
      for i in range(len(total)):
        content = open(oj(redbox,total[i]),'r').readlines()
        if len(content) > 0: 
          if not any([total[i] in l_already for l_already in c_already]):
            notperf.append(redbox+total[i][:-4]+'.jpg'+classification)

    random.shuffle(notperf)
    print "Gathering completed. Pickle dumping it."
    pickle.dump(notperf, open(pickle_fname,'wb'))

  newcomers, notperf_left = notperf[:20000], notperf[20000:]
  pickle.dump(notperf_left, open(pickle_fname,'wb'))
  print "% updated"%(pickle_fname)

  with open(fn_train,'r') as f_train:
    c_train = f_train.readlines()
  c_train += newcomers
  random.shuffle(c_train)
  f_train = open(fn_train,'w')
  f_train.writelines(c_train)















