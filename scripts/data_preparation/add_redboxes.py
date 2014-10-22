#!/usr/bin/env python
import os, random, shutil
from os.path import join as oj
import cPickle as pickle
import subprocess


def bring_all_redbox_positives(task, flag):
  here = os.getcwd()
  os.chdir('/data2/ad6813/caffe/data/'+task)
  cmd = "find /data2/ad6813/pipe-data/Redbox/raw_data/dump/ -name '*.dat' | xargs -i grep -l '"+flag+"' {} | cut -d'.' -f 1 | xargs -i echo '{}.jpg 1' >> train.txt"
  p = subprocess.Popen(cmd, shell=True)
  p.wait()
  os.chdir(here)


def bring_redbox_negatives(task, avoid_flags, classification, add_num, pickle_fname, data_dir, fn_train, using_pickle):

  if os.path.isfile(oj(os.getcwd(),'redbox_vacant_'+task+'_negatives.pickle')) and using_pickle:
    print "Found pickle dump of vacant, non-perfect Redbox images without %s flag. Using it."%(task)
    notperf = pickle.load(open(pickle_fname,'r'))

  else:
    notperf, total = [], []
    for fname in os.listdir(data_dir):
      if fname.endswith('.dat'):
        total.append(fname)

    print 'Gathering vacant, non-perfect Redbox images without %s flag...'%(task)
    with open(fn_train,'r') as f_already:
      c_already = f_already.readlines()
      c_already = [line.split(' ')[0] for line in c_already]
      for i in range(len(total)):
        content = open(oj(data_dir,total[i]),'r').readlines()
        content = [line.strip() for line in content]
        if all([len(content) > 0,
                len([flag for flag in content if flag in avoid_flags])==0,
                total[i] not in c_already]):
          notperf.append(data_dir+total[i][:-4]+'.jpg'+classification+'\n')

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




if __name__ == '__main__':
  import sys

  task = 'scrape'
  avoid_flags = ['NoVisibleEvidenceOfScrapingOrPeeling','PhotoDoesNotShowEnoughOfScrapeZones']
  classification = ' 0'
  using_pickle = True
  add_num = 20000 # how many imgs to add
  pickle_fname = 'redbox_vacant_'+task+'_negatives.pickle'
  data_dir = '/data2/ad6813/pipe-data/Redbox/raw_data/dump/'
  fn_train = '/data2/ad6813/caffe/data/scrape/train.txt'

  bring_redbox_negatives(task, avoid_flags, classification, add_num, pickle_fname, data_dir, fn_train, using_pickle)

  flag = 'NoVisibleEvidenceOfScrapingOrPeeling'
  bring_all_redbox_positives(task, flag)












