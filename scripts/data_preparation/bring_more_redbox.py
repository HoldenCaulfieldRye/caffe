#!/bin/env python

import os, random, shutil


if __name__ == '__main__':

  redbox = '/data2/ad6813/pipe-data/Redbox/raw_data/dump/'
  fn_train = '/data2/ad6813/caffe/data/scrape/train.txt'

  notperf, total = [], []
  for fname in os.listdir(redbox):
    if fname.endswith('.dat'):
      total.append(fname)

  with open(fn_train,'r') as f_already:
    c_already = f_already.readlines()
    for i in range(len(total)):
      content = open(total[i],'r').readlines()
      if len(content) > 0: 
        if not any([total[i] in l_already for l_already in c_already]):
          notperf.append(redbox+total[i][:-4]+'.jpg')
    
  random.shuffle(notperf)
  
  notperf = notperf[:20000]

  with open(fn_train,'r') as f_train:
    c_train = f_train.readlines()
  c_train += notperf
  random.shuffle(c_train)
  f_train = open(fn_train,'w')
  f_train.writelines(c_train)
  shutil.copy(prefix+base+suffix, '/data2/ad6813/pipe-data/CorrRedbox')














