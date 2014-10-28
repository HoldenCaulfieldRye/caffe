#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join as oj
import subprocess

# Usage: python plot.py path/to/model test-inter=.. [start-iter=..] [end-iter==..]

def get_test_interval(model_dir):
  if not os.path.isfile(oj(model_dir,'train_output.log.test')):
    print "ERROR: could not find file named 'train_output.log.test' in", model_dir
    exit
  else:
    test = open(oj(model_dir,'train_output.log.test'),'r').readlines()
    if len(test) == 1:
      print "ERROR: no data in", oj(model_dir,'train_output.log.test')
      exit
    else:  
      return int(test[2].split()[0])
    # return len(open(oj(model_dir,'train_output.log.train'),'r').readlines()) / len(open(oj(model_dir,'train_output.log.test'),'r').readlines()) + 1


def matplot(model_dir, train, val_acc, val_loss, start=-1, end=-1):
  
  if end == start == -1:
    start, end = 0, len(train)
    print 'plotting entire training data'
  
  elif start == -1:
    start = 0
    print 'plotting from iter %i to %i'%(start,end)
    
  elif end == -1:
    print 'plotting from iter %i to the end'%(start)
    end = len(train)

  else:
    print 'plotting from iter %i to %i'%(start,end)

  plt.ylim([0,1.2])
  x = np.array(range(len(train[start:end])))
  ytrain = np.array([float(el[1]) for el in train[start:end]])
  ytest_acc = np.array([float(el[1]) for el in val_acc[start:end]])
  ytest_loss = np.array([np.float(el[1]) for el in val_loss[start:end]])
  # print "\nloss looks like", ytest_loss[:5], ytest_loss[-5:], "\n"
  plt.plot(x, ytrain, label='training loss', color='0.55')
  # plt.plot(x, ytrain, label='training loss')
  if len(x) != len(ytest_acc):
    print 'len(x) %i != %i len(ytrain)'%(len(x),len(ytest_acc))
    sys.exit()
  plt.plot(x, ytest_acc, label='validation accuracy',color='g')
  plt.plot(x, ytest_loss, label='validation loss',color='r')
  plt.legend(loc='upper left')
  plt.xlabel('Iters')
  plt.ylabel('TrainingLoss')
  # plt.title('Go on choose one')
  plt.grid(True)
  plt.savefig(oj(model_dir,'plot_more_'+model_dir.split('/')[-3]+'_'+model_dir.split('/')[-1]+'.png'))
  # plt.show()


def get_caffe_train_errors(model_dir):
  return get_caffe_errors(model_dir,'train',2)

def get_caffe_val_acc(model_dir, test_interval):
  original = get_caffe_errors(model_dir,'test',2)
  stretch = []
  for i in range(len(original)-1):
    for k in range(test_interval):
      stretch.append(original[i])
  stretch.append(original[-1])
  assert all([stretch[i] == stretch[0] for i in range(test_interval)])
  assert stretch[-1] != stretch[-2]
  return stretch

def get_caffe_val_loss(model_dir, test_interval):
  original = get_caffe_errors(model_dir,'test',3)
  stretch = []
  for i in range(len(original)-1):
    for k in range(test_interval):
      stretch.append(original[i])
  stretch.append(original[-1])
  assert all([stretch[i] == stretch[0] for i in range(test_interval)])
  assert stretch[-1] != stretch[-2]
  return stretch

def get_caffe_errors(model_dir, typ, idx):
  data_files = []
  for fname in os.listdir(model_dir):
    if 'train_output' in fname and fname.endswith('.log.'+typ): data_files.append(fname)
  if len(data_files) != 1:
    print 'there is not exactly 1 filename otf \'*train_output*.log.%s\' in given directory'%(typ)
    sys.exit()
  content = open(oj(model_dir,data_files[0]),'r').readlines()
  legit_length = len(content[1])
  content = [' '.join(line.split()).split(' ') for line in content
             if not line.startswith('#')]
  print 'raw content looked like %s and %s'%(content[0], content[-1])

  for i in range(len(content)):
    if len(content[i]) <= idx:
      print 'line[%i] is messed up: %s'%(i,content[i])
      sys.exit()
  content = [(line[0],line[idx]) for line in content]
  # end = len(content)
  # while True:
  #   try:
  #     content = [(line[0],line[idx]) for line in content[:end]]
  #     break
  #   except:
  #     end -= 1
    
  print 'selected content looks like %s and %s'%(content[0], content[-1])
  return content


def parse_log(model_dir):
  cmd = "./parselog.sh "+oj(model_dir,'train_output.log')
  p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  p.wait()


def print_help():
  print """Usage:
    ./plot.py path/to/model [start-epoch=..] [end-epoch=..]"""
  exit


if __name__ == '__main__':

  print('')

  try: 
    os.environ['DISPLAY']
  except: 
    print 'ERROR: X11 forwarding not enabled, cannot run script'
    sys.exit()

  if len(sys.argv) < 2:
      print_help()
  else:
    
    model_dir = os.path.abspath(sys.argv[1])

    parse_log(model_dir)


    start,end = -1,-1
    for arg in sys.argv:
      if arg.startswith("start-iter="):
        start = int(arg.split('=')[-1])
      if arg.startswith("end-iter="):
        end = int(arg.split('=')[-1])

    test_interval = get_test_interval(model_dir)
    train, val_acc, val_loss = get_caffe_train_errors(model_dir), get_caffe_val_acc(model_dir, test_interval), get_caffe_val_loss(model_dir, test_interval)
    print 'train looks like %s and %s'%(train[0], train[-1])
    matplot(model_dir, train, val_acc, val_loss, start, end)

    # ideal would be get layer names from cfg, and prompt for which ones
